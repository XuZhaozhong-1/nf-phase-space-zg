# nfmodel/flows/zg_costh_flow.py
import torch
from torch import nn
from nfmodel.flows.realnvp_nd import RealNVP


class ZGCosthFlow(nn.Module):
    """
    Learn q(c) on (-1,1), where c = cos(theta).
    Internal unconstrained variable: y in R with c = tanh(y).
    RealNVP models y via z <-> y (z ~ N(0,1)).
    """
    def __init__(self, n_blocks=8, hidden=128, permute="reverse", seed=0):
        super().__init__()
        self.flow = RealNVP(dim=1, n_blocks=n_blocks, hidden=hidden, permute=permute, seed=seed)

    # -------- deterministic maps between c and y --------
    @staticmethod
    def c_to_y(c: torch.Tensor):
        """c in (-1,1) -> y in R, shapes: (n,) -> (n,)"""
        eps = 1e-7
        c = torch.clamp(c, -1.0 + eps, 1.0 - eps)
        return 0.5 * torch.log((1.0 + c) / (1.0 - c))  # atanh(c)

    @staticmethod
    def y_to_c(y: torch.Tensor):
        """y in R -> c in (-1,1), accepts (n,) or (n,1); returns (n,)"""
        if y.ndim == 2:
            y = y[:, 0]
        return torch.tanh(y)

    # -------- flow maps between z and y --------
    @torch.no_grad()
    def z_to_y(self, z: torch.Tensor):
        """z -> y via RealNVP forward. z shape (n,1) -> y shape (n,1)"""
        y, _ = self.flow.fwd(z)
        return y

    @torch.no_grad()
    def y_to_z(self, y: torch.Tensor):
        """y -> z via RealNVP inverse. y shape (n,1) -> z shape (n,1)"""
        z, _ = self.flow.inv(y)
        return z

    # -------- combined maps --------
    def sample_c(self, n: int, device="mps" if torch.backends.mps.is_available() else "cpu"):
        """Sample c ~ q_theta(c)."""
        y = self.flow.sample(n, device=device)      # (n,1) ~ q_theta(y)
        return torch.tanh(y[:, 0])                  # (n,)

    @torch.no_grad()
    def z_to_c(self, z: torch.Tensor):
        """Explicit forward: z -> y -> c."""
        y = self.z_to_y(z)                          # (n,1)
        return torch.tanh(y[:, 0])                  # (n,)

    @torch.no_grad()
    def c_to_z(self, c: torch.Tensor):
        """Explicit inverse: c -> y -> z."""
        y = self.c_to_y(c).unsqueeze(-1)            # (n,1)
        z = self.y_to_z(y)                          # (n,1)
        return z

    def logprob_c(self, c: torch.Tensor):
        """
        log q(c) = log q(y) + log|dy/dc|, with y=atanh(c), dy/dc = 1/(1-c^2).
        Returns shape (n,).
        """
        c = torch.clamp(c, -1.0 + 1e-7, 1.0 - 1e-7)
        y = self.c_to_y(c).unsqueeze(-1)            # (n,1)
        logq_y = self.flow.log_prob(y)              # (n,)
        log_abs_dy = -torch.log1p(-c * c)   # -log(1-c^2)
        return logq_y + log_abs_dy