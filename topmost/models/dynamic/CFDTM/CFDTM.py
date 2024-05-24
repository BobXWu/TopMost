import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ETC import ETC
from .UWE import UWE
from topmost.models.Encoder import MLPEncoder


class CFDTM(nn.Module):
    '''
    Modeling Dynamic Topics in Chain-Free Fashion by Evolution-Tracking Contrastive Learning and Unassociated Word Exclusion. ACL 2024 Findings

    Xiaobao Wu, Xinshuai Dong, Liangming Pan, Thong Nguyen, Anh Tuan Luu.
    '''

    def __init__(self,
                 vocab_size,
                 train_time_wordfreq,
                 num_times,
                 pretrained_WE=None,
                 num_topics=50,
                 en_units=100,
                 temperature=0.1,
                 beta_temp=1.0,
                 weight_neg=1.0e+7,
                 weight_pos=1.0e+1,
                 weight_UWE=1.0e+3,
                 neg_topk=15,
                 dropout=0.,
                 embed_size=200
                ):
        super().__init__()

        self.num_topics = num_topics
        self.beta_temp = beta_temp
        self.train_time_wordfreq = train_time_wordfreq
        self.encoder = MLPEncoder(vocab_size, num_topics, en_units, dropout)

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=False)

        if pretrained_WE is None:
            self.word_embeddings  = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size), std=0.1)
            self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        else:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())

        # topic_embeddings: TxKxD
        self.topic_embeddings = nn.init.xavier_normal_(torch.zeros(num_topics, self.word_embeddings.shape[1])).repeat(num_times, 1, 1)
        self.topic_embeddings = nn.Parameter(self.topic_embeddings)

        self.ETC = ETC(num_times, temperature, weight_neg, weight_pos)
        self.UWE = UWE(self.ETC, num_times, temperature, weight_UWE, neg_topk)

    def get_beta(self):
        dist = self.pairwise_euclidean_dist(F.normalize(self.topic_embeddings, dim=-1), F.normalize(self.word_embeddings, dim=-1))
        beta = F.softmax(-dist / self.beta_temp, dim=1)

        return beta

    def pairwise_euclidean_dist(self, x, y):
        cost = torch.sum(x ** 2, axis=-1, keepdim=True) + torch.sum(y ** 2, axis=-1) - 2 * torch.matmul(x, y.t())
        return cost

    def get_theta(self, x, times=None):
        theta, mu, logvar = self.encoder(x)
        if self.training:
            return theta, mu, logvar

        return theta

    def get_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)

        return KLD.mean()

    def get_NLL(self, theta, beta, x, recon_x=None):
        if recon_x is None:
            recon_x = self.decode(theta, beta)
        recon_loss = -(x * recon_x.log()).sum(axis=1)

        return recon_loss

    def decode(self, theta, beta):
        d1 = F.softmax(self.decoder_bn(torch.bmm(theta.unsqueeze(1), beta).squeeze(1)), dim=-1)
        return d1

    def forward(self, x, times):
        loss = 0.

        theta, mu, logvar = self.get_theta(x)
        kl_theta = self.get_KL(mu, logvar)

        loss += kl_theta

        beta = self.get_beta()
        time_index_beta = beta[times]
        recon_x = self.decode(theta, time_index_beta)
        NLL = self.get_NLL(theta, time_index_beta, x, recon_x)
        NLL = NLL.mean()
        loss += NLL

        loss_ETC = self.ETC(self.topic_embeddings)
        loss += loss_ETC

        loss_UWE = self.UWE(self.train_time_wordfreq, beta, self.topic_embeddings, self.word_embeddings)
        loss += loss_UWE

        rst_dict = {
            'loss': loss,
        }

        return rst_dict
