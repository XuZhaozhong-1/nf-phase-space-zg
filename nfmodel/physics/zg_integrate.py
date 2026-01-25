# nfmodel/physics/zg_integrate.py
import numpy as np
from nfmodel.physics.zg_phase_space import build_event_zg, dphi2_dcosth_dphi, MZ_DEFAULT
from nfmodel.physics.cuts import passes_cuts
from nfmodel.physics.zg_me import me2

def integrate_zg_lo(
    n: int,
    Ecm: float,
    mZ: float = MZ_DEFAULT,
    seed: int = 0,
):
    """
    Uniform sampling in (cosθ, φ) with rejection for cuts (reject = weight 0).
    Returns:
      I, dI, sigma_hat, dsigma_hat, acceptance
    """
    rng = np.random.default_rng(seed)
    s = Ecm * Ecm
    flux = 2.0 * s

    jac = dphi2_dcosth_dphi(Ecm, mZ)  # constant for 2->2 chart
    angle_vol = 4.0 * np.pi           # ∫ dcosθ dφ

    # We do a "per-throw" estimator including zeros to keep unbiased with cuts:
    # contrib_i = me2_i if passes else 0
    contrib = np.zeros(n, dtype=np.float64)

    kept = 0
    for i in range(n):
        costh = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0*np.pi)

        p_all = build_event_zg(Ecm, costh, phi, mZ=mZ)

        if not passes_cuts(p_all):
            contrib[i] = 0.0
            continue

        w = me2(p_all)
        if not np.isfinite(w) or w < 0:
            contrib[i] = 0.0
            continue

        contrib[i] = w
        kept += 1

    mean = contrib.mean()
    std = contrib.std(ddof=1) if n > 1 else 0.0
    err_mean = std / np.sqrt(n)

    I = jac * angle_vol * mean
    dI = jac * angle_vol * err_mean

    sigma = I / flux
    dsigma = dI / flux

    acc = kept / n
    return I, dI, sigma, dsigma, acc
