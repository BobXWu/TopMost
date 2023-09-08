"""Poincare ball manifold."""

import torch
from .base import Manifold
from .math_util import TanC, ArTanC


class PoincareBall(Manifold):
    """PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c. (c < 0)

    So that the Poincare ball radius will be 1 / sqrt(-c).
    Notice that the more close c is to 0, the more flat space will be.
    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.truncate_c = lambda x: torch.clamp(x, min=-1e5, max=-1e-5)

    def proj(self, x, c):
        c = self.truncate_c(c)
        x_norm = self.clip(x.norm(dim=-1, keepdim=True, p=2))
        max_norm = (1 - self.eps) / c.abs().sqrt()
        cond = x_norm > max_norm
        projected = x / x_norm * max_norm
        return torch.where(cond, projected, x)

    def proj_tan(self, v, x, c):
        return v

    def proj_tan0(self, v, c):
        return v

    def expmap(self, v, x, c):
        c = self.truncate_c(c)
        v_norm = self.clip(v.norm(p=2, dim=-1, keepdim=True))
        second_term = TanC(self._lambda_x(x, c) * v_norm / 2.0, c) * v / v_norm
        gamma = self._mobius_add(x, second_term, c)
        return gamma

    def expmap0(self, v, c):
        c = self.truncate_c(c)
        v_norm = self.clip(v.norm(p=2, dim=-1, keepdim=True))
        gamma = TanC(v_norm, c) * v / v_norm
        return gamma

    def logmap(self, y, x, c):
        c = self.truncate_c(c)
        sub = self._mobius_add(-x, y, c)
        sub_norm = self.clip(sub.norm(p=2, dim=-1, keepdim=True))
        lam = self._lambda_x(x, c)
        return 2.0 / lam * ArTanC(sub_norm, c) * sub / sub_norm

    def logmap0(self, y, c):
        c = self.truncate_c(c)
        y_norm = self.clip(y.norm(p=2, axis=-1, keepdim=True))
        return ArTanC(y_norm, c) * y / y_norm

    def ptransp(self, v, x, y, c):
        c = self.truncate_c(c)
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, v, c) * lambda_x / lambda_y

    def ptransp0(self, v, x, c):
        c = self.truncate_c(c)
        lambda_x = self._lambda_x(x, c)
        return torch.tensor(2.0, dtype=self.dtype) * v / lambda_x

    def dist(self, x, y, c):
        c = self.truncate_c(c)
        return 2.0 * ArTanC(self._mobius_add(-x, y, c).norm(p=2, dim=-1), c)

    def egrad2rgrad(self, grad, x, c):
        c = self.truncate_c(c)
        metric = torch.square(self._lambda_x(x, c))
        return grad / metric

    def inner(self, v1, v2, x, c, keep_shape=False):
        c = self.truncate_c(c)
        metric = torch.square(self._lambda_x(x, c))
        product = v1 * metric * v2
        res = product.sum(dim=-1, keepdim=True)
        if keep_shape:
            # return tf.broadcast_to(res, x.shape)
            last_dim = x.shape.as_list()[-1]
            return torch.cat([res for _ in range(last_dim)], dim=-1)
        return torch.squeeze(res, dim=-1)

    def retraction(self, v, x, c):
        c = self.truncate_c(c)
        new_v = self.expmap(v, x, c)
        return self.proj(new_v, c)

    def _mobius_add(self, x, y, c):
        c = self.truncate_c(c)
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * c * xy - c * y2) * x + (1 + c * x2) * y
        denom = 1 - 2 * c * xy + (c ** 2) * x2 * y2
        return num / self.clip(denom)

    def _mobius_mul(self, x, a, c):
        c = self.truncate_c(c)
        x_norm = self.clip(x.norm(p=2, dim=-1, keepdim=True))
        scale = TanC(a * ArTanC(x_norm, c), c) / x_norm
        return scale * x

    def _mobius_matvec(self, x, a, c):
        c = self.truncate_c(c)
        x_norm = self.clip(x.norm(p=2, dim=-1, keepdim=True))
        mx = torch.matmul(x, a)
        mx_norm = self.clip(mx.norm(p=2, dim=-1, keepdim=True))
        res = TanC(mx_norm / x_norm * ArTanC(x_norm, c), c) * mx / mx_norm
        return res

    def _lambda_x(self, x, c):
        c = self.truncate_c(c)
        x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
        return self.clip(2.0 / (1.0 + c * x_sqnorm))

    def _gyration(self, x, y, v, c):
        xy = self._mobius_add(x, y, c)
        yv = self._mobius_add(y, v, c)
        xyv = self._mobius_add(x, yv, c)
        return self._mobius_add(-xy, xyv, c)
