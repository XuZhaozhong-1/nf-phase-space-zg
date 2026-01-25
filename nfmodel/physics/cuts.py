# nfmodel/physics/cuts.py
import numpy as np

def pT(p):
    # p = (E, px, py, pz)
    return np.sqrt(p[1]**2 + p[2]**2)

def eta(p):
    E, px, py, pz = p
    return 0.5 * np.log((E + pz) / (E - pz + 1e-12))

def phi(p):
    return np.arctan2(p[2], p[1])

def deltaR(p1, p2):
    dphi = np.arctan2(np.sin(phi(p1) - phi(p2)), np.cos(phi(p1) - phi(p2)))
    deta = eta(p1) - eta(p2)
    return np.sqrt(deta*deta + dphi*dphi)


def passes_cuts(p_all):
    """
    Cuts for q qbar -> Z g  (Z + 1 jet)

    Expected event layout (like MG5 standalone):
      p_all shape: (4,4)
        p_all[0] = beam1
        p_all[1] = beam2
        p_all[2] = Z
        p_all[3] = g (jet)

    Returns True if passes cuts.
    """
    p_all = np.asarray(p_all)
    if p_all.shape != (4, 4):
        return False
    if not np.all(np.isfinite(p_all)):
        return False

    Z = p_all[2]
    g = p_all[3]

    # basic positivity
    if Z[0] <= 0 or g[0] <= 0:
        return False

    # ---- typical LO-safe cuts ----
    # jet pT cut
    if pT(g) < 20.0:
        return False

    # jet eta cut
    if abs(eta(g)) > 5.0:
        return False

    # optional: Z pT / eta cuts (often not required, but can be used)
    # if pT(Z) < 0.0: return False
    # if abs(eta(Z)) > 10.0: return False

    return True
