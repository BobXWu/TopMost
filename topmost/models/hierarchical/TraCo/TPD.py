import torch
from torch import nn
from . import utils


class TPD(nn.Module):
    def __init__(self, sinkhorn_alpha, sinkhorn_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, topic_embeddings_list, weight_loss_TPD=20.0):
        all_loss_TPD = 0.
        transp_list = list()

        num_layers = len(topic_embeddings_list)

        for layer_id in range(num_layers)[:-1]:
            topic_embeddings = topic_embeddings_list[layer_id]
            next_topic_embeddings = topic_embeddings_list[layer_id + 1]
            cost = utils.pairwise_euclidean_distance(topic_embeddings, next_topic_embeddings)
            loss_TPD, transp = self.sinkhorn(cost, return_transp=True)

            all_loss_TPD += loss_TPD
            transp_list.append(transp)

        all_loss_TPD *= weight_loss_TPD / (num_layers - 1)

        return loss_TPD, transp_list

    def sinkhorn(self, M, return_transp=False):
        device = M.device

        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.sinkhorn_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss = torch.sum(transp * M)

        if return_transp:
            return loss, transp

        return loss
