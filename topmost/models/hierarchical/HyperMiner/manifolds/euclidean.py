"""Euclidean manifold."""

import torch
from .base import Manifold


class Euclidean(Manifold):
    """Euclidean Manifold class. Usually we refer it as R^n.

    Attributes:
        name (str): The manifold name, and its value is "Euclidean".
    """

    def __init__(self, **kwargs):
        """Initialize an Euclidean manifold.
        Args:
            **kwargs: Description
        """
        super(Euclidean, self).__init__(**kwargs)
        self.name = 'Euclidean'

    def proj(self, x, c):
        return x

    def proj_tan(self, v, x, c):
        return v

    def proj_tan0(self, v, c):
        return v

    def expmap(self, v, x, c):
        return v + x

    def expmap0(self, v, c):
        return v

    def logmap(self, y, x, c):
        return y - x

    def logmap0(self, y, c):
        return y

    def ptransp(self, v, x, y, c):
        return torch.ones_like(x) * v

    def ptransp0(self, v, x, c):
        return torch.ones_like(x) * v

    def dist(self, x, y, c):
        sqdis = torch.sum((x - y).pow(2), dim=-1)
        return sqdis.sqrt()

    def egrad2rgrad(self, grad, x, c):
        return grad

    def inner(self, v1, v2, x, c, keep_shape=False):
        if keep_shape:
            # In order to keep the same computation logic in Ada* Optimizer
            return v1 * v2
        else:
            return torch.sum(v1 * v2, dim=-1, keepdim=False)
