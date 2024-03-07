import torch


def get_topk_tensor(matrix, topk, return_mask=False):
    topk_values, topk_idx = torch.topk(matrix, topk)
    topk_tensor = torch.zeros_like(matrix)
    if return_mask:
        topk_tensor.scatter_(1, topk_idx, 1)
    else:
        topk_tensor.scatter_(1, topk_idx, topk_values)

    return topk_tensor


def pairwise_euclidean_distance(x, y):
    cost = torch.sum(x ** 2, axis=-1, keepdim=True) + torch.sum(y ** 2, dim=-1) - 2 * torch.matmul(x, y.t())
    return cost
