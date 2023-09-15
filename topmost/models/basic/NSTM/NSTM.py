

import torch
from torch import nn
import torch.nn.functional as F
from .auto_diff_sinkhorn import sinkhorn_loss


class NSTM(nn.Module):
    '''
        Neural Topic Model via Optimal Transport. ICLR 2021

        He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray Buntine.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.25, pretrained_WE=None, train_WE=True, embed_size=200, recon_loss_weight=0.07, sinkhorn_alpha=20):
        super().__init__()

        self.recon_loss_weight = recon_loss_weight
        self.sinkhorn_alpha = sinkhorn_alpha

        self.e1 = nn.Linear(vocab_size, en_units)
        self.e2 = nn.Linear(en_units, num_topics)
        self.e_dropout = nn.Dropout(dropout)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(self.topic_embeddings)

    def get_beta(self):
        word_embedding_norm = F.normalize(self.word_embeddings)
        topic_embedding_norm = F.normalize(self.topic_embeddings)
        beta = torch.matmul(topic_embedding_norm, word_embedding_norm.T)
        return beta

    def get_theta(self, input):
        theta = F.relu(self.e1(input))
        theta = self.e_dropout(theta)
        theta = self.mean_bn(self.e2(theta))
        theta = F.softmax(theta, dim=-1)
        return theta

    def forward(self, input):
        theta = self.get_theta(input)
        beta = self.get_beta()
        M = 1 - beta
        sh_loss = sinkhorn_loss(M, theta.T, F.softmax(input, dim=-1).T, lambda_sh=self.sinkhorn_alpha)
        recon = F.softmax(torch.matmul(theta, beta), dim=-1)
        recon_loss = -(input * recon.log()).sum(axis=1)

        loss = self.recon_loss_weight * recon_loss + sh_loss
        loss = loss.mean()
        return {'loss': loss}
