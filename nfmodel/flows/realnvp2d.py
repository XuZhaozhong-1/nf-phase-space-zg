# nfmodel/flows/realnvp2d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)

def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    # z: (..., D)
    return -0.5 * (z**2 + LOG_2PI).sum(dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class AffineCoupling(nn.Module):
    """
    RealNVP affine coupling for 2D with a binary mask.
    y = x_masked + (1-mask)*( x*exp(s(masked)) + t(masked) )
    """
    def __init__(self, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)          # shape (2,)
        self.st = MLP(in_dim=2, out_dim=2*2, hidden=hidden)

    def forward(self, x):
        # x: (B,2)
        x_masked = x * self.mask
        st = self.st(x_masked)  # (B,4)
        s, t = st[:, :2], st[:, 2:]
        # only apply to unmasked dim(s)
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(dim=-1)
        return y, logdet

    def inverse(self, y):
        y_masked = y * self.mask
        st = self.st(y_masked)
        s, t = st[:, :2], st[:, 2:]
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = (-s).sum(dim=-1)
        return x, logdet

class RealNVP2D(nn.Module):
    def __init__(self, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        masks = []
        # alternate masks: [1,0] and [0,1]
        for i in range(n_layers):
            if i % 2 == 0:
                masks.append(torch.tensor([1.0, 0.0]))
            else:
                masks.append(torch.tensor([0.0, 1.0]))
        self.layers = nn.ModuleList([AffineCoupling(m, hidden=hidden) for m in masks])

    def fwd(self, z):
        # z -> y, returns y and total logdet
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet = logdet + ld
        return x, logdet

    def inv(self, y):
        # y -> z
        x = y
        logdet = torch.zeros(y.shape[0], device=y.device)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld
        return x, logdet

    def sample_y(self, n: int, device="cpu"):
        z = torch.randn(n, 2, device=device)
        y, _ = self.fwd(z)
        return y

    def logprob_y(self, y):
        # log q(y)
        z, logdet = self.inv(y)
        return standard_normal_logprob(z) + logdet
