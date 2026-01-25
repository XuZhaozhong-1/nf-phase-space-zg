# nfmodel/flows/zg_costh_flow.py
import torch
from torch import nn
from nfmodel.flows.realnvp_nd import RealNVP

class ZGCosthFlow(nn.Module):
    """
    Learn q(costh) on (-1,1).
    Internally: y in R, costh = tanh(y).

    Provides:
      sample_costh(n) -> costh in (-1,1)
      logprob_costh(costh) -> log q(costh)
    """
    def __init__(self, n_blocks=8, hidden=128, permute="reverse", seed=0):
        super().__init__()
        # dim=1 => permutations are no-op, but we keep the interface consistent
        self.flow = RealNVP(dim=1, n_blocks=n_blocks, hidden=hidden, permute=permute, seed=seed)

    def sample_costh(self, n, device="cpu"):
        y = self.flow.sample(n, device=device)      # (n,1) in R
        costh = torch.tanh(y[:, 0])                 # (n,) in (-1,1)
        return costh

    def logprob_costh(self, costh: torch.Tensor):
        # costh: (n,)
        # y = atanh(costh), and q(costh)=q(y)*|dy/dcosth|
        eps = 1e-7
        costh = torch.clamp(costh, -1.0 + eps, 1.0 - eps)
        y = 0.5 * torch.log((1.0 + costh) / (1.0 - costh))   # atanh
        y = y.unsqueeze(-1)                                   # (n,1)

        logq_y = self.flow.log_prob(y)                         # (n,)

        # dy/dcosth = 1/(1-costh^2)
        log_abs_dy = -torch.log1p(-costh * costh + 1e-12)

        return logq_y + log_abs_dy