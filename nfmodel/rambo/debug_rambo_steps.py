import numpy as np


# -----------------------------
# Stage 1: U -> raw massless 4-vectors
# -----------------------------
def raw_from_U(U):
    """
    U: shape (n,4) uniform in [0,1]
    returns p_raw: shape (n,4) with (E, px, py, pz), massless by construction.
    """
    U = np.asarray(U)
    n = U.shape[0]

    costh = 2 * U[:, 0] - 1
    sinth = np.sqrt(np.maximum(0.0, 1.0 - costh**2))
    phi   = 2 * np.pi * U[:, 1]

    # Energy distribution used in your code
    E = -np.log(np.clip(U[:, 2] * U[:, 3], 1e-300, 1.0))

    p = np.zeros((n, 4), dtype=np.float64)
    p[:, 0] = E
    p[:, 1] = E * sinth * np.cos(phi)
    p[:, 2] = E * sinth * np.sin(phi)
    p[:, 3] = E * costh
    return p


# -----------------------------
# Stage 2: boost to CM (sum spatial momentum -> 0)
# -----------------------------
def boost_to_cm(p):
    """
    Boost all 4-vectors so that the total 3-momentum is zero.
    Uses the same algebra as your rambo_from_U().
    """
    p = np.asarray(p, dtype=np.float64).copy()

    Q = p.sum(axis=0)
    Q0, Qx, Qy, Qz = Q

    # boost velocity beta = -Qvec / Q0
    bx = -Qx / Q0
    by = -Qy / Q0
    bz = -Qz / Q0

    b2 = bx*bx + by*by + bz*bz
    if b2 >= 1.0:
        raise RuntimeError(f"Unphysical boost: |beta|^2={b2} (should be < 1)")

    gamma = 1.0 / np.sqrt(1.0 - b2)
    gamma2 = (gamma - 1.0) / b2 if b2 > 0.0 else 0.0

    for i in range(p.shape[0]):
        E_i, px, py, pz = p[i]
        bp = bx*px + by*py + bz*pz

        px_new = px + gamma2*bp*bx + gamma*bx*E_i
        py_new = py + gamma2*bp*by + gamma*by*E_i
        pz_new = pz + gamma2*bp*bz + gamma*bz*E_i
        E_new  = gamma*(E_i + bp)

        p[i] = [E_new, px_new, py_new, pz_new]

    return p


# -----------------------------
# Stage 3: rescale so sum(E) = Ecm
# -----------------------------
def rescale_to_Ecm(p, Ecm):
    p = np.asarray(p, dtype=np.float64).copy()
    sumE = p[:, 0].sum()
    if sumE <= 0:
        raise RuntimeError(f"Bad sumE={sumE}")
    scale = Ecm / sumE
    return p * scale


# -----------------------------
# Diagnostics
# -----------------------------
def report_numbers(tag, p):
    """
    p: shape (n,4) with (E, px, py, pz)
    Reports:
      - sumE
      - |sum p|
      - Q^2 = (sumE)^2 - |sum p|^2
      - max/mean |m_i^2| where m_i^2 = E_i^2 - |p_i|^2
    """
    p = np.asarray(p, dtype=np.float64)

    E  = p[:, 0]
    px = p[:, 1]
    py = p[:, 2]
    pz = p[:, 3]

    sumE = E.sum()
    sump = np.array([px.sum(), py.sum(), pz.sum()], dtype=np.float64)
    sump_mag = np.linalg.norm(sump)

    m2 = E**2 - (px**2 + py**2 + pz**2)
    Q2 = sumE**2 - sump_mag**2

    print(f"\n[{tag}]")
    print(f"  sumE           = {sumE:.12e}")
    print(f"  sumPx,Py,Pz    = ({sump[0]:.12e}, {sump[1]:.12e}, {sump[2]:.12e})")
    print(f"  |sum p|        = {sump_mag:.12e}")
    print(f"  Q^2            = {Q2:.12e}")
    print(f"  max |m_i^2|    = {np.max(np.abs(m2)):.12e}")
    print(f"  mean |m_i^2|   = {np.mean(np.abs(m2)):.12e}")


def main(Ecm=1000.0, n=3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    U = np.random.rand(n, 4)

    p_raw = raw_from_U(U)
    report_numbers("RAW (from U)", p_raw)

    p_boost = boost_to_cm(p_raw)
    report_numbers("AFTER BOOST (CM)", p_boost)

    p_final = rescale_to_Ecm(p_boost, Ecm)
    report_numbers("AFTER RESCALE (sumE=Ecm)", p_final)


if __name__ == "__main__":
    main(Ecm=1000.0, n=3, seed=123)
