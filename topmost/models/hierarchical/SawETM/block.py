import torch.nn as nn


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise RuntimeError("activation should be relu/tanh/softplus, not {}".format(activation))


class ResBlock(nn.Module):
    """Simple MLP block with residual connection.

    Args:
        in_features: the feature dimension of each output sample.
        out_features: the feature dimension of each output sample.
        activation: the activation function of intermediate layer, relu or gelu.
    """

    def __init__(self, in_features, out_features, activation="relu"):
        super(ResBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)

        self.bn = nn.BatchNorm1d(out_features)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        if self.in_features == self.out_features:
            out = self.fc2(self.activation(self.fc1(x)))
            return self.activation(self.bn(x + out))
        else:
            x = self.fc1(x)
            out = self.fc2(self.activation(x))
            return self.activation(self.bn(x + out))
