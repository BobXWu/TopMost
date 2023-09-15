

import torch
import torch.nn as nn
import torch.nn.functional as F


class ETM(nn.Module):
    '''
        Topic Modeling in Embedding Spaces. TACL 2020

        Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei.
    '''
    def __init__(self, vocab_size, embed_size=200, num_topics=50, en_units=800, dropout=0., pretrained_WE=None, train_WE=False):
        super().__init__()

        if pretrained_WE is not None:
            self.word_embeddings = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.word_embeddings = nn.Parameter(torch.randn((vocab_size, embed_size)))

        self.word_embeddings.requires_grad = train_WE

        self.topic_embeddings = nn.Parameter(torch.randn((num_topics, self.word_embeddings.shape[1])))

        self.encoder1 = nn.Sequential(
            nn.Linear(vocab_size, en_units),
            nn.ReLU(),
            nn.Linear(en_units, en_units),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        e1 = self.encoder1(x)
        return self.fc21(e1), self.fc22(e1)

    def get_theta(self, x):
        # Warn: normalize the input if use Relu.
        # https://github.com/adjidieng/ETM/issues/3
        norm_x = x / x.sum(1, keepdim=True)
        mu, logvar = self.encode(norm_x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)
        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def get_beta(self):
        beta = F.softmax(torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=1)
        return beta

    def forward(self, x, avg_loss=True):
        theta, mu, logvar = self.get_theta(x)
        beta = self.get_beta()
        recon_x = torch.matmul(theta, beta)

        loss = self.loss_function(x, recon_x, mu, logvar, avg_loss)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar, avg_loss=True):
        recon_loss = -(x * (recon_x + 1e-12).log()).sum(1)
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        loss = (recon_loss + KLD)

        if avg_loss:
            loss = loss.mean()

        return loss
