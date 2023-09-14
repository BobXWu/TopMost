import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NMTM(nn.Module):
    '''
        Learning Multilingual Topics with Neural Variational Inference. NLPCC 2020.

        Xiaobao Wu, Chunping Li, Yan Zhu, Yishu Miao.
    '''
    def __init__(self, Map_en2cn, Map_cn2en, vocab_size_en, vocab_size_cn, num_topics=50, en_units=200, dropout=0., lam=0.8):
        super().__init__()

        self.num_topics = num_topics
        self.lam = lam

        # V_en x V_cn
        self.Map_en2cn = nn.Parameter(torch.as_tensor(Map_en2cn).float(), requires_grad=False)

        # V_cn x V_en
        self.Map_cn2en = nn.Parameter(torch.as_tensor(Map_cn2en).float(), requires_grad=False)

        self.a = 1 * np.ones((1, int(num_topics))).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False)
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T), requires_grad=False)

        self.decoder_bn_en = nn.BatchNorm1d(vocab_size_en, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn_en.weight.requires_grad = False

        self.decoder_bn_cn = nn.BatchNorm1d(vocab_size_cn, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn_cn.weight.requires_grad = False

        self.fc11_en = nn.Linear(vocab_size_en, en_units)
        self.fc11_cn = nn.Linear(vocab_size_cn, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

        self.fc1_drop = nn.Dropout(dropout)
        self.z_drop = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn.weight.requires_grad = False

        self.phi_en = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_en))))
        self.phi_cn = nn.Parameter(nn.init.xavier_uniform_(torch.empty((num_topics, vocab_size_cn))))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x, lang):
        e1 = F.softplus(getattr(self, f'fc11_{lang}')(x))

        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        theta = self.reparameterize(mu, logvar)
        theta = F.softmax(theta, dim=1)
        theta = self.z_drop(theta)
        return theta, mu, logvar

    def get_theta(self, x, lang):
        theta, mu, logvar = self.encode(x, lang)

        if self.training:
            return theta, mu, logvar
        else:
            return mu

    def get_beta(self):
        beta_en = self.lam * torch.matmul(self.phi_cn, self.Map_cn2en) + (1 - self.lam) * self.phi_en
        beta_cn = self.lam * torch.matmul(self.phi_en, self.Map_en2cn) + (1 - self.lam) * self.phi_cn
        return beta_en, beta_cn

    def decode(self, theta, lang):
        d1 = F.softmax(getattr(self, f'decoder_bn_{lang}')(torch.matmul(theta, getattr(self, f'beta_{lang}'))), dim=1)
        return d1

    def forward(self, x_en, x_cn):
        self.beta_en, self.beta_cn = self.get_beta()

        theta_en, mu_en, logvar_en = self.get_theta(x_en, lang='en')
        theta_cn, mu_cn, logvar_cn = self.get_theta(x_cn, lang='cn')

        x_recon_en = self.decode(theta_en, lang='en')
        x_recon_cn = self.decode(theta_cn, lang='cn')

        loss_en = self.loss_function(x_recon_en, x_en, mu_en, logvar_en)
        loss_cn = self.loss_function(x_recon_cn, x_cn, mu_cn, logvar_cn)

        loss = loss_en + loss_cn

        rst_dict = {
            'loss': loss
        }

        return rst_dict

    def loss_function(self, recon_x, x, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        RECON = -(x * (recon_x + 1e-10).log()).sum(1)

        LOSS = (RECON + KLD).mean()
        return LOSS
