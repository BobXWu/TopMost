
import torch
import torch.nn as nn
import torch.nn.functional as F
from .TopicDistQuant import TopicDistQuant
from .TSC import TSC


class TSCTM(nn.Module):
    '''
        Mitigating Data Sparsity for Short Text Topic Modeling by Topic-Semantic Contrastive Learning. EMNLP 2022

        Xiaobao Wu, Anh Tuan Luu, Xinshuai Dong.

        Note: This implementation does not include TSCTM with augmentations. For augmentations, see https://github.com/BobXWu/TSCTM.
    '''

    def __init__(self, vocab_size, num_topics=50, en_units=200, temperature=0.5, weight_contrast=1.0):
        super().__init__()

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.topic_dist_quant = TopicDistQuant(num_topics, num_topics)
        self.contrast_loss = TSC(temperature, weight_contrast)

    def get_beta(self):
        return self.fcd1.weight.T

    def encode(self, inputs):
        e1 = F.softplus(self.fc11(inputs))
        e1 = F.softplus(self.fc12(e1))
        return self.mean_bn(self.fc21(e1))

    def decode(self, theta):
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def get_theta(self, inputs):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)
        return softmax_theta

    def forward(self, inputs):
        theta = self.encode(inputs)
        softmax_theta = F.softmax(theta, dim=1)

        quant_rst = self.topic_dist_quant(softmax_theta)

        recon = self.decode(quant_rst['quantized'])
        loss = self.loss_function(recon, inputs) + quant_rst['loss']

        features = torch.cat([F.normalize(theta, dim=1).unsqueeze(1)], dim=1)
        contrastive_loss = self.contrast_loss(features, quant_idx=quant_rst['encoding_indices'])
        loss += contrastive_loss

        return {'loss': loss, 'contrastive_loss': contrastive_loss}

    def loss_function(self, recon_x, x):
        loss = -(x * (recon_x).log()).sum(axis=1)
        loss = loss.mean()
        return loss
