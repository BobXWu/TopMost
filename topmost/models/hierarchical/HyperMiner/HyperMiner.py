
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import manifolds
from ..SawETM.SawETM import SawETM


class HyperMiner(SawETM):
    """
        HyperMiner: Topic Taxonomy Mining with Hyperbolic Embedding. NeurIPS 2022.

        Yishi Xu, Dongsheng Wang, Bo Chen, Ruiying Lu, Zhibin Duan, Mingyuan Zhou.

        https://github.com/NoviceStone/HyperMiner
    """
    def __init__(self, vocab_size, num_topics_list, device='cpu', manifold="PoincareBall", clip_r=None, curvature=-0.01, embed_size=50, hidden_size=300, pretrained_WE=None):
        super().__init__(vocab_size, num_topics_list, device, embed_size, hidden_size, pretrained_WE)

        self.manifold = getattr(manifolds, manifold)()

        if curvature is not None:
            self.curvature = torch.tensor([curvature])
            self.curvature = self.curvature.to(device)
        else:
            self.curvature = nn.Parameter(torch.Tensor([-1.]))

        self.clip_r = clip_r

    def feat_clip(self, x):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        cond = x_norm > self.clip_r
        projected = x / x_norm * self.clip_r
        return torch.where(cond, projected, x)

    @property
    def bottom_word_embeddings(self):
        return self.rho

    @property
    def topic_embeddings_list(self):
        return self.alpha[::-1]

    def get_phi(self):
        """Returns the factor loading matrix by utilizing sawtooth connection.
        """
        phis = []
        for n in range(self.num_layers):
            if n == 0:
                hyp_rho = self.manifold.proj(
                    self.manifold.expmap0(self.rho, self.curvature), self.curvature)
                hyp_alpha = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n], self.curvature), self.curvature)
                phi = torch.softmax(-self.manifold.dist(
                    hyp_rho.unsqueeze(1), hyp_alpha.unsqueeze(0), self.curvature), dim=0)
            else:
                hyp_alpha1 = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n - 1], self.curvature), self.curvature)
                hyp_alpha2 = self.manifold.proj(
                    self.manifold.expmap0(self.alpha[n], self.curvature), self.curvature)
                phi = torch.softmax(-self.manifold.dist(
                    hyp_alpha1.unsqueeze(1).detach(), hyp_alpha2.unsqueeze(0), self.curvature), dim=0)
            phis.append(phi)
        return phis

    def get_beta(self):
        beta_list = list()
        phis = self.get_phi()
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
        phis = self.get_phi()
        phis = phis[1:]
        return [item.T for item in phis][::-1]

    def get_theta(self, x):
        hidden_feats = []
        for n in range(self.num_layers):
            if n == 0:
                hidden_feats.append(self.h_encoder[n](x))
            else:
                hidden_feats.append(self.h_encoder[n](hidden_feats[-1]))

        phis = self.get_phi()

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
