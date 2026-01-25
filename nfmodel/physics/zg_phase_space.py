# nfmodel/physics/zg_phase_space.py
import numpy as np

MZ_DEFAULT = 91.1876

def build_event_zg(Ecm: float, costh: float, phi: float, mZ: float = MZ_DEFAULT) -> np.ndarray:
    """
    Partonic CM frame: q qbar -> Z g

    Returns p_all with shape (4,4):
      [0]=beam1, [1]=beam2, [2]=Z, [3]=g
    and columns [E, px, py, pz]
    """
    s = Ecm * Ecm
    if s <= mZ * mZ:
        raise ValueError("Below threshold: Ecm^2 <= mZ^2")

    # Incoming (massless), along z
    p1 = np.array([Ecm/2.0, 0.0, 0.0, +Ecm/2.0], dtype=np.float64)
    p2 = np.array([Ecm/2.0, 0.0, 0.0, -Ecm/2.0], dtype=np.float64)

    # 2-body momentum magnitude for masses (mZ, 0)
    p = (s - mZ*mZ) / (2.0 * Ecm)

    sinth = np.sqrt(max(0.0, 1.0 - costh*costh))
    px = p * sinth * np.cos(phi)
    py = p * sinth * np.sin(phi)
    pz = p * costh

    Eg = p
    EZ = np.sqrt(p*p + mZ*mZ)

    pZ = np.array([EZ,  px,  py,  pz], dtype=np.float64)
    pg = np.array([Eg, -px, -py, -pz], dtype=np.float64)  # recoil

    return np.vstack([p1, p2, pZ, pg])


def dphi2_dcosth_dphi(Ecm: float, mZ: float = MZ_DEFAULT) -> float:
    """
    dPhi2 = [1/(16*pi^2)] * (|p|/Ecm) * dOmega
    with dOmega = dcos(theta) dphi.
    """
    s = Ecm * Ecm
    if s <= mZ*mZ:
        return 0.0
    p = (s - mZ*mZ) / (2.0 * Ecm)
    return (1.0 / (16.0 * np.pi**2)) * (p / Ecm)
