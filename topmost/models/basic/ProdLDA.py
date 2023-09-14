

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProdLDA(nn.Module):
    '''
        Autoencoding Variational Inference For Topic Models. ICLR 2017

        Akash Srivastava, Charles Sutton.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.4):
        super().__init__()

        self.num_topics = num_topics

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

        # align with the default parameters of tf.contrib.layers.batch_norm
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/batch_norm
        # center=True (add bias(beta)), scale=False (weight(gamma) is not used)
        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(num_topics))
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn.weight.data.copy_(torch.ones(num_topics))
        self.logvar_bn.weight.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))
        self.decoder_bn.weight.requires_grad = False

        self.fc1_drop = nn.Dropout(dropout)
        self.theta_drop = nn.Dropout(dropout)

        self.fcd1 = nn.Linear(num_topics, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.fcd1.weight)

    def get_beta(self):
        return self.fcd1.weight.T

    def get_theta(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        theta = self.theta_drop(theta)
        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        return self.mean_bn(self.fc21(e1)), self.logvar_bn(self.fc22(e1))

    def decode(self, theta):
        d1 = F.softmax(self.decoder_bn(self.fcd1(theta)), dim=1)
        return d1

    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)
        recon_x = self.decode(theta)
        loss = self.loss_function(x, recon_x, mu, logvar)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        loss = (recon_loss + KLD).mean()
        return loss
