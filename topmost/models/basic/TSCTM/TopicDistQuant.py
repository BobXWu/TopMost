
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopicDistQuant(nn.Module):
    '''
        Short Text Topic Modeling with Topic Distribution Quantization and Negative Sampling Decoder. EMNLP 2020

        Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao
    '''
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.copy_(torch.eye(embedding_dim))
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Calculate distances
        # NOTE: Do not use torch.cdist. It has unknown bugs.
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                   + torch.sum(self._embedding.weight**2, dim=1)
                   - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize and unflatten
        quantized = self._embedding(encoding_indices)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='none').sum(axis=1).mean()
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='none').sum(axis=1).mean()
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        rst = {
            'loss': loss,
            'quantized': quantized,
            'encoding_indices': encoding_indices,
        }

        return rst
