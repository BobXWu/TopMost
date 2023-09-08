"""Math utils functions."""

import torch
from torch import tan, atan, cos, acos, sin, asin
from torch import tanh, atanh, cosh, acosh, sinh, asinh


def Tan(x):
    """Computes tangent of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor. Has the same type as x.
    """
    return tan(x)


def Tanh(x):
    """Computes hyperbolic tangent of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type as x.
    """
    return tanh(torch.clamp(x, min=-15, max=15))


def TanC(x, c):
    """A unified tangent and inverse tangent function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (tensor): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return 1 / c.abs().sqrt() * Tanh(x * c.abs().sqrt())  # c < 0


def ArTan(x):
    """Computes inverse tangent of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type as x.
    """
    return atan(torch.clamp(x, min=-15, max=15))


def ArTanh(x):
    """Computes inverse hyperbolic tangent of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type as x.
    """
    return atanh(torch.clamp(x, min=-1 + 1e-7, max=1 - 1e-7))


def ArTanC(x, c):
    """A unified hyperbolic tangent and inverse hyperbolic tangent function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (tensor): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return 1 / c.abs().sqrt() * ArTanh(x * c.abs().sqrt())  # c < 0


def Cos(x):
    """Computes cosine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return cos(x)


def Cosh(x):
    """Computes hyperbolic cosine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return cosh(torch.clamp(x, min=-15, max=15))


def ArCos(x):
    """Computes inverse cosine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return acos(torch.clamp(x, min=-1 + 1e-7, max=1 - 1e-7))


def ArCosh(x):
    """Computes inverse hyperbolic cosine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return acosh(torch.clamp(x, min=1 + 1e-7, max=1e15))


def Sin(x):
    """Computes sine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return sin(x)


def Sinh(x):
    """Computes hyperbolic sine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return sinh(torch.clamp(x, min=-15, max=15))


def SinC(x, c):
    """A unified sine and inverse sine function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (tensor): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return 1 / c.abs().sqrt() * Sinh(x * c.abs().sqrt())  # c < 0


def ArSin(x):
    """Computes inverse sine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return asin(torch.clamp(x, min=-1 + 1e-7, max=1 - 1e-7))


def ArSinh(x):
    """Computes inverse hyperbolic sine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return asinh(torch.clamp(x, min=-15, max=15))


def ArSinC(x, c):
    """A unified hyperbolic sine and inverse hyperbolic sine function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (tensor): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return 1 / c.abs().sqrt() * ArSinh(x * c.abs().sqrt())  # c < 0
