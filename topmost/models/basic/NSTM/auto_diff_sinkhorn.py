import torch


def sinkhorn_loss(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2):
    device = a.device

    u = (torch.ones_like(a) / a.size()[0]).to(device)
    # TODO v is zeros in the tensorflow code.
    # v = (torch.ones_like(b)).to(device)

    K = torch.exp(-M * lambda_sh)
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t()))) 
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))  
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)

    return sinkhorn_divergences
