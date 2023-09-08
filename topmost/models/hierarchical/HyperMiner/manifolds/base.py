"""Base manifold."""

import torch


class Manifold(object):
    """Abstract class to define basic operations on a manifold.

    Attributes:
        clip (function): Clips tensor values to a specified min and max.
        dtype: The type of the variables.
        eps (float): A small constant value.
        max_norm (float): The maximum value for number clipping.
        min_norm (float): The minimum value for number clipping.
    """

    def __init__(self, **kwargs):
        """Initialize a manifold.
        """
        super(Manifold, self).__init__()

        self.min_norm = 1e-15
        self.max_norm = 1e15
        self.eps = 1e-5

        self.dtype = kwargs["dtype"] if "dtype" in kwargs else torch.float32
        self.clip = lambda x: torch.clamp(x, min=self.min_norm, max=self.max_norm)

    def proj(self, x, c):
        """A projection function that prevents x from leaving the manifold.
        Args:
            x (tensor): A point should be on the manifold, but it may not meet the manifold constraints.
            c (tensor): The manifold curvature.
        Returns:
            tensor: A projected point, meeting the manifold constraints.
        """
        raise NotImplementedError

    def proj_tan(self, v, x, c):
        """A projection function that prevents v from leaving the tangent space of point x.
        Args:
            v (tensor): A point should be on the tangent space, but it may not meet the manifold constraints.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """
        raise NotImplementedError

    def proj_tan0(self, v, c):
        """A projection function that prevents v from leaving the tangent space of origin point.
        Args:
            v (tensor): A point should be on the tangent space, but it may not meet the manifold constraints.
            c (tensor): The manifold curvature.
        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """
        raise NotImplementedError

    def expmap(self, v, x, c):
        """Map a point v in the tangent space of point x to the manifold.
        Args:
            v (tensor): A point in the tangent space of point x.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        raise NotImplementedError

    def expmap0(self, v, c):
        """Map a point v in the tangent space of origin point to the manifold.
        Args:
            v (tensor): A point in the tangent space of origin point.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        raise NotImplementedError

    def logmap(self, y, x, c):
        """Map a point y on the manifold to the tangent space of x.
        Args:
            y (tensor): A point on the manifold.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of mapping y to the tangent space of x.
        """
        raise NotImplementedError

    def logmap0(self, y, c):
        """Map a point y on the manifold to the tangent space of origin point.
        Args:
            y (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of mapping y to the tangent space of origin point.
        """
        raise NotImplementedError

    def ptransp(self, v, x, y, c):
        """Parallel transport function, used to move point v in the tangent space of x to the tangent space of y.
        Args:
            v (tensor): A point in the tangent space of x.
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of transporting v from the tangent space at x to the tangent space at y.
        """
        raise NotImplementedError

    def ptransp0(self, v, x, c):
        """Parallel transport function, used to move point v in the tangent space of origin point to the tangent space of y.
        Args:
            v (tensor): A point in the tangent space of origin point.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of transporting v from the tangent space at origin point to the tangent space at y.
        """
        raise NotImplementedError

    def dist(self, x, y, c):
        """Calculate the squared geodesic/distance between x and y.
        Args:
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: the geodesic/distance between x and y.
        """
        raise NotImplementedError

    def egrad2rgrad(self, grad, x, c):
        """Computes Riemannian gradient from the Euclidean gradient, typically used in Riemannian optimizers.
        Args:
            grad (tensor): Euclidean gradient at x.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: Riemannian gradient at x.
        """
        raise NotImplementedError

    def inner(self, v1, v2, x, c, keep_shape):
        """Computes the inner product of a pair of tangent vectors v1 and v2 at x.
        Args:
            v1 (tensor): A tangent point at x.
            v2 (tensor): A tangent point at x.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
            keep_shape (bool, optional): Whether the output tensor keeps shape or not.
        Returns:
            tensor: The inner product of v1 and v2 at x.
        """
        raise NotImplementedError

    def retraction(self, v, x, c):
        """Retraction is a continuous map function from tangent space to the manifold, typically used in Riemannian optimizers.
        The exp map is one of retraction functions.
        Args:
            v (tensor): A tangent point at x.
            x (tensor): A point on the manifold.
            c (tensor): The manifold curvature.
        Returns:
            tensor: The result of mapping tangent point v at x to the manifold.
        """
        return self.proj(self.expmap(v, x, c), c)
