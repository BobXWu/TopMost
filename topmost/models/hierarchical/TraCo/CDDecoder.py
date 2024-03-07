import torch
from torch import nn
import torch.nn.functional as F
from . import utils


class CDDecoder(nn.Module):
    def __init__(self, num_layers, vocab_size, bias_p, bias_topk):
        super().__init__()
        self.num_layers = num_layers
        self.bias_p = bias_p
        self.bias_topk = bias_topk

        self.decoder_bn_list = nn.ModuleList([nn.BatchNorm1d(vocab_size, affine=False) for _ in range(num_layers)])

        self.bias_vectors = nn.ParameterList([])
        for _ in range(num_layers):
            bias_vector = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, vocab_size)))
            self.bias_vectors.append(bias_vector)

    def forward(self, input_bow, theta_list, beta_list):
        topk_bias_list = list()
        all_recon_loss = 0.

        for layer_id in range(self.num_layers):
            topk_bias = utils.get_topk_tensor(beta_list[layer_id], topk=self.bias_topk).sum(0)
            topk_bias = topk_bias.detach()
            topk_bias_list.append(topk_bias)

        for layer_id in range(self.num_layers):
            topk_bias = 0.
            # previous layer
            if layer_id > 0:
                topk_bias += topk_bias_list[layer_id - 1]

            # next layer
            if layer_id < self.num_layers - 1:
                topk_bias += topk_bias_list[layer_id + 1]

            topk_mask = (topk_bias > 0).float()
            bias = self.bias_p * topk_bias * topk_mask + self.bias_vectors[layer_id] * (1 - topk_mask)

            recon = self.decoder_bn_list[layer_id](torch.matmul(theta_list[layer_id], beta_list[layer_id]))
            recon = recon + bias
            recon = F.softmax(recon, dim=-1)
            recon_loss = -(input_bow * (recon + 1e-12).log()).sum(axis=1)
            recon_loss = recon_loss.mean()
            all_recon_loss += recon_loss

        all_recon_loss /= self.num_layers

        return all_recon_loss
