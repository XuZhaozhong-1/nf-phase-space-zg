import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# Utility: convert torch/numpy/list → numpy safely
# ============================================================

def to_numpy(x):
    """
    Safely convert a tensor/array/list to a NumPy array.
    Handles:
        - numpy.ndarray  (returned directly)
        - torch.Tensor   (.detach().cpu().numpy())
        - python lists   (np.array)
    """
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


# ============================================================
# Ensure output directory exists
# ============================================================

def ensure_plot_dir():
    if not os.path.exists("plots"):
        os.makedirs("plots")


# ============================================================
# 1. Training loss curve
# ============================================================

def plot_loss_curve(steps, losses):
    ensure_plot_dir()

    steps = to_numpy(steps)
    losses = to_numpy(losses)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Weighted NLL")
    plt.title("NF Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/loss_curve.png", dpi=200)
    plt.close()

    print("[plot_all] Saved: plots/loss_curve.png")


# ============================================================
# 2. U-space marginals (RAMBO × |M|² vs NF)
# ============================================================

def plot_U_marginals(xs_rambo, ws_rambo, xs_nf, n_plot=6):
    ensure_plot_dir()

    # Convert everything to numpy
    xs_r = to_numpy(xs_rambo)
    ws_r = to_numpy(ws_rambo)
    xs_n = to_numpy(xs_nf)

    D = xs_r.shape[1]

    for i in range(min(n_plot, D)):
        plt.figure(figsize=(6, 4))

        # RAMBO target density
        plt.hist(
            xs_r[:, i],
            bins=50,
            weights=ws_r,
            density=True,
            alpha=0.4,
            label="RAMBO × |M|²",
        )

        # NF samples
        plt.hist(
            xs_n[:, i],
            bins=50,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="NF samples",
        )

        plt.xlabel(f"x[{i}]")
        plt.ylabel("Density")
        plt.title(f"U-space marginal {i}")
        plt.legend()
        plt.tight_layout()

        fname = f"plots/U_marginal_{i}.png"
        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"[plot_all] Saved: {fname}")


# ============================================================
# Physics helpers (copied from your physics code)
# ============================================================

def pT(p):
    return np.sqrt(p[...,1]**2 + p[...,2]**2)

def eta(p):
    return 0.5 * np.log((p[0] + p[3]) / (p[0] - p[3] + 1e-12))

def deltaR(p1, p2):
    phi1 = np.arctan2(p1[2], p1[1])
    phi2 = np.arctan2(p2[2], p2[1])
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    deta = eta(p1) - eta(p2)
    return np.sqrt(deta**2 + dphi**2)


# ============================================================
# 3. Physics-space plots from NF samples
# ============================================================

def plot_physics_distributions(xs_nf, Ecm=1000.0, n_plot=5000):
    """
    xs_nf : NF-generated x samples (flattened 12D)
    Converts x → p_final → plots pT, eta, ΔR distributions.
    """
    ensure_plot_dir()

    # Convert to numpy
    xs_nf = to_numpy(xs_nf)

    # Only use first few thousand
    xs_nf = xs_nf[: min(n_plot, xs_nf.shape[0])]

    # storage
    pT_m, pT_p, pT_g = [], [], []
    eta_m, eta_p, eta_g = [], [], []
    dR_mumu, dR_mg, dR_pg = [], [], []

    # Loop over events
    for x in xs_nf:
        # reshape 12 → (3,4)
        final = x.reshape(3, 4)

        # incoming beams
        p1 = np.array([Ecm/2, 0, 0, +Ecm/2])
        p2 = np.array([Ecm/2, 0, 0, -Ecm/2])
        mu_m, mu_p, g = final

        # compute observables
        pT_m.append(pT(mu_m))
        pT_p.append(pT(mu_p))
        pT_g.append(pT(g))

        eta_m.append(eta(mu_m))
        eta_p.append(eta(mu_p))
        eta_g.append(eta(g))

        dR_mumu.append(deltaR(mu_m, mu_p))
        dR_mg.append(deltaR(mu_m, g))
        dR_pg.append(deltaR(mu_p, g))

    # -------------------------
    # pT plots
    # -------------------------
    def hist_plot(data, label, fname):
        plt.figure(figsize=(6,4))
        plt.hist(data, bins=50, density=True, alpha=0.7, label=label)
        plt.xlabel(label)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"plots/{fname}", dpi=200)
        plt.close()
        print(f"[plot_all] Saved: plots/{fname}")

    hist_plot(pT_m, "pT(mu-)", "pT_mu_minus.png")
    hist_plot(pT_p, "pT(mu+)", "pT_mu_plus.png")
    hist_plot(pT_g, "pT(gluon)", "pT_gluon.png")

    # -------------------------
    # eta plots
    # -------------------------
    hist_plot(eta_m, "eta(mu-)", "eta_mu_minus.png")
    hist_plot(eta_p, "eta(mu+)", "eta_mu_plus.png")
    hist_plot(eta_g, "eta(gluon)", "eta_gluon.png")

    # -------------------------
    # DeltaR
    # -------------------------
    hist_plot(dR_mumu, "ΔR(mu-, mu+)", "dR_mumu.png")
    hist_plot(dR_mg, "ΔR(mu-, g)", "dR_mug.png")
    hist_plot(dR_pg, "ΔR(mu+, g)", "dR_pug.png")