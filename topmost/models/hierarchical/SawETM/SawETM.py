
import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import ResBlock


class SawETM(nn.Module):
    """
        Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network. ICML 2021.

        Zhibin Duan, Dongsheng Wang, Bo Chen, Chaojie Wang, Wenchao Chen, Yewen Li, Jie Ren, Mingyuan Zhou.

        https://github.com/ZhibinDuan/SawETM
    """
    def __init__(self, vocab_size, num_topics_list, device='cpu', embed_size=100, hidden_size=256, pretrained_WE=None):
        super().__init__()
        # constants
        self.device = device
        self.gam_prior = torch.tensor(1.0, dtype=torch.float, device=self.device)
        self.real_min = torch.tensor(1e-30, dtype=torch.float, device=self.device)
        self.theta_max = torch.tensor(1000.0, dtype=torch.float, device=self.device)
        self.wei_shape_min = torch.tensor(1e-1, dtype=torch.float, device=self.device)
        self.wei_shape_max = torch.tensor(100.0, dtype=torch.float, device=self.device)

        # hyper-parameters
        self.num_topics_list = num_topics_list[::-1]
        self.num_hiddens_list = [hidden_size] * len(self.num_topics_list)

        assert len(self.num_topics_list) == len(self.num_hiddens_list)
        self.num_layers = len(self.num_topics_list)

        # learnable word embeddings
        if pretrained_WE is not None:
            self.rho = nn.Parameter(torch.from_numpy(pretrained_WE).float())
        else:
            self.rho = nn.Parameter(torch.empty(vocab_size, embed_size).normal_(std=0.02))

        # topic embeddings for different latent layers
        self.alpha = nn.ParameterList([])
        for n in range(self.num_layers):
            self.alpha.append(nn.Parameter(
                torch.empty(self.num_topics_list[n], embed_size).normal_(std=0.02)))

        # deterministic mapping to obtain hierarchical features
        self.h_encoder = nn.ModuleList([])
        for n in range(self.num_layers):
            if n == 0:
                self.h_encoder.append(
                    ResBlock(vocab_size, self.num_hiddens_list[n]))
            else:
                self.h_encoder.append(
                    ResBlock(self.num_hiddens_list[n - 1], self.num_hiddens_list[n]))

        # variational encoder to obtain posterior parameters
        self.q_theta = nn.ModuleList([])
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                self.q_theta.append(
                    nn.Linear(self.num_hiddens_list[n], 2 * self.num_topics_list[n]))
            else:
                self.q_theta.append(nn.Linear(
                    self.num_hiddens_list[n] + self.num_topics_list[n], 2 * self.num_topics_list[n]))

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min))

    def reparameterize(self, shape, scale, sample_num=50):
        """Returns a sample from a Weibull distribution via reparameterization.
        """
        shape = shape.unsqueeze(0).repeat(sample_num, 1, 1)
        scale = scale.unsqueeze(0).repeat(sample_num, 1, 1)
        eps = torch.rand_like(shape, dtype=torch.float, device=self.device)
        samples = scale * torch.pow(- self.log_max(1 - eps), 1 / shape)
        return torch.clamp(samples.mean(0), self.real_min.item(), self.theta_max.item())

    def kl_weibull_gamma(self, wei_shape, wei_scale, gam_shape, gam_scale):
        """Returns the Kullback-Leibler divergence between a Weibull distribution and a Gamma distribution.
        """
        euler_mascheroni_c = torch.tensor(0.5772, dtype=torch.float, device=self.device)
        t1 = torch.log(wei_shape) + torch.lgamma(gam_shape)
        t2 = - gam_shape * torch.log(wei_scale * gam_scale)
        t3 = euler_mascheroni_c * (gam_shape / wei_shape - 1) - 1
        t4 = gam_scale * wei_scale * torch.exp(torch.lgamma(1 + 1 / wei_shape))
        return (t1 + t2 + t3 + t4).sum(1).mean()

    def get_nll(self, x, x_reconstruct):
        """Returns the negative Poisson likelihood of observational count data.
        """
        log_likelihood = self.log_max(x_reconstruct) * x - torch.lgamma(1.0 + x) - x_reconstruct
        neg_log_likelihood = - torch.sum(log_likelihood, dim=1, keepdim=False).mean()
        return neg_log_likelihood

    @property
    def bottom_word_embeddings(self):
        return self.rho

    @property
    def topic_embeddings_list(self):
        return self.alpha[::-1]

    def get_phis(self):
        """Returns the factor loading matrix by utilizing sawtooth connection.
        """
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                phi = torch.softmax(torch.mm(self.rho, self.alpha[n].transpose(0, 1)), dim=0)
            else:
                phi = torch.softmax(torch.mm(self.alpha[n - 1].detach(), self.alpha[n].transpose(0, 1)), dim=0)
            phis.append(phi)
        return phis

    def get_beta(self):
        beta_list = list()
        phis = self.get_phis()
        last_beta  = None

        for layer_id, phi in enumerate(phis):
            if layer_id == 0:
                last_beta = phi.T
            else:
                last_beta = torch.matmul(phi.T, last_beta)
            beta_list.append(last_beta)

        beta_list = beta_list[::-1]
        return beta_list

    def get_phi_list(self):
        phis = self.get_phis()
        phis = phis[1:]
        return [item.T for item in phis][::-1]

    def get_theta(self, x):
        hidden_feats = []
        for n in range(self.num_layers):
            if n == 0:
                hidden_feats.append(self.h_encoder[n](x))
            else:
                hidden_feats.append(self.h_encoder[n](hidden_feats[-1]))

        # =================================================================================
        # phis:
        # 5000x50
        # 50x36
        # 36x12
        # 12x2
        phis = self.get_phis()

        ks = []
        lambs = []
        thetas = []
        phi_by_theta_list = []
        for n in range(self.num_layers - 1, -1, -1):
            if n == self.num_layers - 1:
                joint_feat = hidden_feats[n]
            else:
                joint_feat = torch.cat((hidden_feats[n], phi_by_theta_list[0]), dim=1)

            k, lamb = torch.chunk(F.softplus(self.q_theta[n](joint_feat)), 2, dim=1)
            k = torch.clamp(k, self.wei_shape_min.item(), self.wei_shape_max.item())
            lamb = torch.clamp(lamb, self.real_min.item())

            if self.training:
                lamb = lamb / torch.exp(torch.lgamma(1 + 1 / k))
                theta = self.reparameterize(k, lamb, sample_num=3) if n == 0 else self.reparameterize(k, lamb)
            else:
                theta = torch.min(lamb, self.theta_max)

            phi_by_theta = torch.mm(theta, phis[n].t())
            phi_by_theta_list.insert(0, phi_by_theta)
            thetas.insert(0, theta)
            lambs.insert(0, lamb)
            ks.insert(0, k)

        if self.training:
            return ks, lambs, phi_by_theta_list, thetas
        else:
            return thetas[::-1]

    def forward(self, x):
        """Forward pass: compute the kl loss and data likelihood.
        """

        ks, lambs, phi_by_theta_list, thetas = self.get_theta(x)

        # =================================================================================
        nll = self.get_nll(x, phi_by_theta_list[0])

        kl_loss = []
        for n in range(self.num_layers):
            if n == self.num_layers - 1:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], self.gam_prior, self.gam_prior))
            else:
                kl_loss.append(self.kl_weibull_gamma(
                    ks[n], lambs[n], phi_by_theta_list[n + 1], self.gam_prior))

        nelbo = nll + sum(kl_loss)

        return {'loss': nelbo}
