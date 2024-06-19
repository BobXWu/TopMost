
import torch
from torch import nn
import torch.nn.functional as F


class DETM(nn.Module):
    """
        The Dynamic Embedded Topic Model. 2019

        Adji B. Dieng, Francisco J. R. Ruiz, David M. Blei
    """
    def __init__(self, vocab_size, num_times, train_size, train_time_wordfreq, num_topics=50, train_WE=True, pretrained_WE=None, en_units=800, eta_hidden_size=200, rho_size=300, enc_drop=0.0, eta_nlayers=3, eta_dropout=0.0, delta=0.005, theta_act='relu', device='cpu'):
        super().__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.num_times = num_times
        self.vocab_size = vocab_size
        self.eta_hidden_size = eta_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.eta_nlayers = eta_nlayers
        self.t_drop = nn.Dropout(enc_drop)
        self.eta_dropout = eta_dropout
        self.delta = delta
        self.train_WE = train_WE
        self.train_size = train_size
        self.rnn_inp = train_time_wordfreq
        self.device = device

        self.theta_act = self.get_activation(theta_act)

        ## define the word embedding matrix \rho
        if self.train_WE:
            self.rho = nn.Linear(self.rho_size, self.vocab_size, bias=False)
        else:
            rho = nn.Embedding(pretrained_WE.size())
            rho.weight.data = torch.from_numpy(pretrained_WE)
            self.rho = rho.weight.data.clone().float().to(self.device)

        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size))

        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
            nn.Linear(self.vocab_size + self.num_topics, en_units), 
            self.theta_act,
            nn.Linear(en_units, en_units),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(en_units, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(en_units, self.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, dropout=self.eta_dropout)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.num_topics, self.num_topics, bias=True)

        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

    def get_activation(self, act):
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'softplus': nn.Softplus(),
            'rrelu': nn.RReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'glu': nn.GLU(),
        }

        if act in activations:
            act = activations[act]
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_alpha(self): ## mean field
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        # TODO: why logsigma_p_0 is zero?
        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 

            p_mu_t = alphas[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()

    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(self.device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)

        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()

        return etas, kl_eta

    def get_theta(self, bows, times, eta=None): ## amortized inference
        """Returns the topic proportions.
        """

        normalized_bows = bows / bows.sum(1, keepdims=True)

        if eta is None and self.training is False:
            eta, kl_eta = self.get_eta(self.rnn_inp)

        eta_td = eta[times]
        inp = torch.cat([normalized_bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))

        if self.training:
            return theta, kl_theta
        else:
            return theta

    @property
    def word_embeddings(self):
        return self.rho.weight

    @property
    def topic_embeddings(self):
        alpha, _ = self.get_alpha()
        return alpha

    def get_beta(self, alpha=None):
        """Returns the topic matrix \beta of shape T x K x V
        """

        if alpha is None and self.training is False:
            alpha, kl_alpha = self.get_alpha()

        if self.train_WE:
            logit = self.rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1)

        beta = F.softmax(logit, dim=-1)

        return beta

    def get_NLL(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = torch.log(loglik + 1e-12)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll

    def forward(self, bows, times):
        bsz = bows.size(0)
        coeff = self.train_size / bsz
        eta, kl_eta = self.get_eta(self.rnn_inp)
        theta, kl_theta = self.get_theta(bows, times, eta)
        kl_theta = kl_theta.sum() * coeff

        alpha, kl_alpha = self.get_alpha()
        beta = self.get_beta(alpha)

        beta = beta[times]
        # beta = beta[times.type('torch.LongTensor')]
        nll = self.get_NLL(theta, beta, bows)
        nll = nll.sum() * coeff

        loss = nll + kl_eta + kl_theta

        rst_dict = {
            'loss': loss,
            'nll': nll,
            'kl_eta': kl_eta,
            'kl_theta': kl_theta
        }

        loss += kl_alpha
        rst_dict['kl_alpha'] = kl_alpha

        return rst_dict

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \\eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
