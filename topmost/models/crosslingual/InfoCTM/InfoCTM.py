import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from topmost.models.Encoder import MLPEncoder
from .TAMI import TAMI


class InfoCTM(nn.Module):
    '''
        InfoCTM: A Mutual Information Maximization Perspective of Cross-lingual Topic Modeling. AAAI 2023

        Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu
    '''
    def __init__(self, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn, vocab_size_en, vocab_size_cn, num_topics=50, en_units=200, dropout=0., temperature=0.2, pos_threshold=0.4, weight_MI=30.0):
        super().__init__()

        self.num_topics = num_topics

        self.encoder_en = MLPEncoder(vocab_size_en, num_topics, en_units, dropout)
        self.encoder_cn = MLPEncoder(vocab_size_cn, num_topics, en_units, dropout)

        self.a = 1 * np.ones((1, int(num_topics))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_en = nn.BatchNorm1d(vocab_size_en, affine=True)
        self.decoder_bn_en.weight.requires_grad = False
        self.decoder_bn_cn = nn.BatchNorm1d(vocab_size_cn, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_cn))))

        self.TAMI = TAMI(temperature, weight_MI, pos_threshold, trans_e2c, pretrain_word_embeddings_en, pretrain_word_embeddings_cn)

    def get_beta(self):
        beta_en = self.phi_en
        beta_cn = self.phi_cn
        return beta_en, beta_cn

    def get_theta(self, x, lang):
        theta, mu, logvar = getattr(self, f'encoder_{lang}')(x)

        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def decode(self, theta, beta, lang):
        bn = getattr(self, f'decoder_bn_{lang}')
        d1 = F.softmax(bn(torch.matmul(theta, beta)), dim=1)
        return d1

    def forward(self, x_en, x_cn):
        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')

        beta_en, beta_cn = self.get_beta()

        loss = 0.

        x_recon_en = self.decode(theta_en, beta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, beta_cn, lang='cn')
        loss_en = self.compute_loss_TM(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.compute_loss_TM(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn

        fea_en = beta_en.T
        fea_cn = beta_cn.T
        loss_TAMI = self.TAMI(fea_en, fea_cn)

        loss += loss_TAMI

        rst_dict = {
            'loss': loss,
            'loss_TAMI': loss_TAMI
        }

        return rst_dict

    def compute_loss_TM(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS
