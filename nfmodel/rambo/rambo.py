import numpy as np
from nfmodel.physics.cuts import passes_cuts
from nfmodel.physics.zllg_me import me_val

def rambo_from_U(U, Ecm):
    """
    RAMBO map: U -> n massless final-state momenta.

    U : shape (n,4) in [0,1]
    Returns:
       p : shape (n,4)
    """
    U = np.asarray(U)
    n = U.shape[0]

    # angular variables
    costh = 2*U[:,0] - 1
    sinth = np.sqrt(1 - costh**2)
    phi   = 2*np.pi*U[:,1]

    # energy variable
    E = -np.log(U[:,2] * U[:,3])

    # raw 4-vectors
    p = np.zeros((n,4))
    p[:,0] = E
    p[:,1] = E * sinth * np.cos(phi)
    p[:,2] = E * sinth * np.sin(phi)
    p[:,3] = E * costh

    # total momentum before boost
    Q = p.sum(axis=0)
    Q0, Qx, Qy, Qz = Q
    Qvec2 = Qx**2 + Qy**2 + Qz**2

    # boost to zero momentum
    bx = -Qx / Q0
    by = -Qy / Q0
    bz = -Qz / Q0
    b2 = bx*bx + by*by + bz*bz
    gamma = 1.0 / np.sqrt(1 - b2)
    gamma2 = (gamma - 1.0) / b2 if b2 > 0 else 0.0

    for i in range(n):
        E_i, px, py, pz = p[i]
        bp = bx*px + by*py + bz*pz

        px_new = px + gamma2*bp*bx + gamma*bx*E_i
        py_new = py + gamma2*bp*by + gamma*by*E_i
        pz_new = pz + gamma2*bp*bz + gamma*bz*E_i
        E_new  = gamma*(E_i + bp)

        p[i] = [E_new, px_new, py_new, pz_new]

    # rescale energies
    scale = Ecm / p[:,0].sum()
    p *= scale

    return p


def random_rambo_event(Ecm=1000.0, n=3):
    """
    Returns:
      p_all : (5,4)
      U     : (3,4)
    """
    U = np.random.rand(n,4)
    p_final = rambo_from_U(U, Ecm)

    # incoming beams
    p1 = np.array([Ecm/2,0,0,+Ecm/2])
    p2 = np.array([Ecm/2,0,0,-Ecm/2])

    p_all = np.vstack([p1, p2, p_final])
    return p_all, U

def rambo_mc_estimate(n_mc=20000, Ecm=1000.0):
    """
    Plain RAMBO Monte Carlo estimate of <|M|^2> after cuts.
    Used as baseline to compare importance sampling.
    """
    ws = []
    accepted = 0

    for _ in range(n_mc):
        # generate a RAMBO event
        p_all, U = random_rambo_event(Ecm=Ecm)

        if not passes_cuts(p_all):
            continue

        w = me_val(p_all)
        if w <= 0 or not np.isfinite(w):
            continue

        ws.append(w)
        accepted += 1

    ws = np.array(ws)
    mean = ws.mean()
    err  = ws.std(ddof=1) / np.sqrt(len(ws))

    print(f"[rambo_mc_estimate] Accepted {accepted}/{n_mc}")
    return mean, err