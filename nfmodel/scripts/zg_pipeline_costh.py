# nfmodel/scripts/zg_pipeline_costh.py
import math
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from nfmodel.flows.zg_costh_flow import ZGCosthFlow
from nfmodel.physics.zg_phase_space import build_event_zg, dphi2_dcosth_dphi, MZ_DEFAULT
from nfmodel.physics.cuts import passes_cuts
from nfmodel.physics.zg_me import me2

TWOPI = 2.0 * math.pi

# Project root: .../NF_Model (robust to os.chdir in zg_me.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -----------------------
# Utils
# -----------------------
def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sample_uniform_costh(n, rng):
    return rng.uniform(-1.0, 1.0, size=n)


def sample_uniform_phi(n, rng):
    return rng.uniform(0.0, TWOPI, size=n)


def m2_value(Ecm, mZ, costh, phi):
    """Compute M^2 with cuts; return 0 if cut fail or invalid."""
    p_all = build_event_zg(Ecm, float(costh), float(phi), mZ=mZ)
    if not passes_cuts(p_all):
        return 0.0
    val = me2(p_all)
    if val is None:
        return 0.0
    if not np.isfinite(val) or val < 0.0:
        return 0.0
    return float(val)


# -----------------------
# TRAIN (cosθ only)
# -----------------------
@torch.no_grad()
def compute_weights_costh_batch(costh_t: torch.Tensor, Ecm: float, mZ: float):
    """
    Exact weights for 2->2 unpolarized scattering:
    w_i ∝ |M|^2(costh_i)
    """
    n = costh_t.shape[0]
    w = torch.zeros(n, dtype=torch.float32)

    phi = 0.0  # arbitrary, φ-independence guaranteed for 2->2 in CM

    for i in range(n):
        w[i] = float(m2_value(Ecm, mZ, float(costh_t[i].cpu()), phi))

    return w.to(costh_t.device)


def train_flow_costh(
    Ecm=1000.0,
    mZ=MZ_DEFAULT,
    steps=2500,
    batch_size=5096,
    lr=2e-4,
    n_blocks=8,
    hidden=16,
    permute="reverse",
    seed=0,
    model_path=None,
):
    device = pick_device()
    print(f"[train] device={device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    flow = ZGCosthFlow(n_blocks=n_blocks, hidden=hidden, permute=permute, seed=seed).to(device)
    opt = optim.Adam(flow.parameters(), lr=lr)

    losses = []
    best = float("inf")

    for step in range(1, steps + 1):
        costh = 2.0 * torch.rand(batch_size, device=device) - 1.0

        w = compute_weights_costh_batch(costh, Ecm=Ecm, mZ=mZ)
        ws = w.sum()
        if ws <= 0:
            continue
        w = w / ws

        logq = flow.logprob_costh(costh)
        loss = -(w * logq).sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        lval = float(loss.item())
        losses.append(lval)
        best = min(best, lval)

        if step % 100 == 0:
            frac = float((w > 0).float().mean().item())
            print(f"[train] step {step:5d}  loss {lval:.6f}  best {best:.6f}  w_nonzero {frac:.3f}")

    if model_path is not None:
        out = Path(model_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": flow.state_dict(),
                "n_blocks": n_blocks,
                "hidden": hidden,
                "permute": permute,
                "seed": seed,
                "Ecm": Ecm,
                "mZ": mZ,
            },
            out,
        )
        print(f"[train] saved: {out.resolve()}")

    return flow, losses


# -----------------------
# INTEGRATION
# -----------------------
def integrate_baseline_uniform(Ecm, mZ, n=20000, seed=0):
    """Uniform in (cosθ, φ)."""
    rng = np.random.default_rng(seed)
    s = Ecm * Ecm
    flux = 2.0 * s
    jac = dphi2_dcosth_dphi(Ecm, mZ)

    costh = sample_uniform_costh(n, rng)
    phi = sample_uniform_phi(n, rng)

    contrib = np.zeros(n, dtype=np.float64)
    for i in range(n):
        contrib[i] = m2_value(Ecm, mZ, costh[i], phi[i])

    mean = contrib.mean()
    err = contrib.std(ddof=1) / np.sqrt(n)

    # Uniform sampling estimates ∫ dcos dphi M^2 as 4π * mean
    I = jac * (4.0 * math.pi) * mean
    dI = jac * (4.0 * math.pi) * err

    sigma = I / flux
    dsigma = dI / flux
    return I, dI, sigma, dsigma


@torch.no_grad()
def integrate_nf_costh(flow: ZGCosthFlow, Ecm, mZ, n=20000, seed=0):
    """
    Importance sampling with:
      cosθ ~ q(cosθ) from flow
      φ ~ Uniform(0,2π) independently
    Weight for angular integral: M^2 * (2π) / q(cosθ)
    """
    rng = np.random.default_rng(seed)
    device = next(flow.parameters()).device

    s = Ecm * Ecm
    flux = 2.0 * s
    jac = dphi2_dcosth_dphi(Ecm, mZ)

    costh = flow.sample_costh(n, device=device)      # (n,)
    logq = flow.logprob_costh(costh)                # (n,)
    q = torch.exp(logq).detach().cpu().numpy()
    costh_np = costh.detach().cpu().numpy()

    phi = sample_uniform_phi(n, rng)

    contrib = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = m2_value(Ecm, mZ, costh_np[i], phi[i])
        if q[i] <= 0 or not np.isfinite(q[i]):
            contrib[i] = 0.0
        else:
            contrib[i] = val * (TWOPI / q[i])

    mean = contrib.mean()
    err = contrib.std(ddof=1) / np.sqrt(n)

    # contrib estimates ∫ dcos dphi M^2 directly
    I = jac * mean
    dI = jac * err

    sigma = I / flux
    dsigma = dI / flux
    return I, dI, sigma, dsigma


# -----------------------
# PLOTTING
# -----------------------
def plot_training_curve(losses, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Training loss (cosθ flow)")
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=150)
    plt.close()


def plot_histograms(flow, Ecm, mZ, out_dir: Path, n_target=120000, n_nf=120000, seed=0):
    """
    Compare marginals:
      Target: uniform (cosθ,φ) weighted by M^2
      NF: cosθ sampled from flow, φ uniform (so φ should be flat)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Target samples: uniform angles + weights
    costh_u = sample_uniform_costh(n_target, rng)
    phi_u = sample_uniform_phi(n_target, rng)
    w = np.zeros(n_target, dtype=np.float64)
    for i in range(n_target):
        w[i] = m2_value(Ecm, mZ, costh_u[i], phi_u[i])
    w = w / (w.sum() + 1e-30)

    # NF samples
    device = next(flow.parameters()).device
    costh_nf = flow.sample_costh(n_nf, device=device).detach().cpu().numpy()
    phi_nf = sample_uniform_phi(n_nf, rng)

    # cosθ histogram
    bins_costh = np.linspace(-1.0, 1.0, 61)
    tgt_costh, _ = np.histogram(costh_u, bins=bins_costh, weights=w)
    nf_costh, _ = np.histogram(costh_nf, bins=bins_costh)
    nf_costh = nf_costh / (nf_costh.sum() + 1e-30)
    centers_costh = 0.5 * (bins_costh[:-1] + bins_costh[1:])

    plt.figure()
    plt.plot(centers_costh, tgt_costh, label=r"Target $\propto |\mathcal{M}|^2$ (weighted)")
    plt.plot(centers_costh, nf_costh, label=r"NF $q(\cos\theta)$ samples")
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel("normalized probability per bin")
    plt.title(r"Marginal over $\cos\theta$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hist_costh_target_vs_nf.png", dpi=150)
    plt.close()

    # phi histogram
    bins_phi = np.linspace(0.0, TWOPI, 61)
    tgt_phi, _ = np.histogram(phi_u, bins=bins_phi, weights=w)
    nf_phi, _ = np.histogram(phi_nf, bins=bins_phi)
    nf_phi = nf_phi / (nf_phi.sum() + 1e-30)
    centers_phi = 0.5 * (bins_phi[:-1] + bins_phi[1:])

    plt.figure()
    plt.plot(centers_phi, tgt_phi, label=r"Target $\propto |\mathcal{M}|^2$ (weighted)")
    plt.plot(centers_phi, nf_phi, label=r"NF $\phi \sim \mathrm{Uniform}(0,2\pi)$")
    plt.xlabel(r"$\phi$")
    plt.ylabel("normalized probability per bin")
    plt.title(r"Marginal over $\phi$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hist_phi_target_vs_nf.png", dpi=150)
    plt.close()


# -----------------------
# SAVE RESULTS
# -----------------------
def save_comparison(out_path: Path, baseline: dict, nf: dict, meta: dict):
    dIb = float(baseline["dI"])
    dIn = float(nf["dI"])
    vr = (dIb / dIn) ** 2 if dIn > 0 else float("inf")

    record = {
        "baseline": baseline,
        "nf_costh": nf,
        "variance_reduction": vr,
        "meta": meta,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[saved] comparison → {out_path.resolve()}")


# -----------------------
# MAIN PIPELINE
# -----------------------
def main():
    Ecm = 1000.0
    mZ = MZ_DEFAULT

    # Train settings
    steps = 2500
    batch_size = 5096
    hidden = 16
    n_blocks = 8

    # Eval settings
    n_eval = 20000

    model_path = PROJECT_ROOT / "nfmodel" / "models" / "zg_costh_flow.pt"
    plot_dir = PROJECT_ROOT / "nfmodel" / "plots" / "zg_pipeline_costh"
    results_path = PROJECT_ROOT / "nfmodel" / "results" / "zg_costh_comparison.json"

    # 1) Train
    flow, losses = train_flow_costh(
        Ecm=Ecm,
        mZ=mZ,
        steps=steps,
        batch_size=batch_size,
        hidden=hidden,
        n_blocks=n_blocks,
        model_path=model_path,
    )

    # 2) Evaluate baseline vs NF (with timing)
    t0 = time.time()
    Ib, dIb, sb, dsb = integrate_baseline_uniform(Ecm, mZ, n=n_eval, seed=1)
    t1 = time.time()
    In, dIn, sn, dsn = integrate_nf_costh(flow, Ecm, mZ, n=n_eval, seed=2)
    t2 = time.time()

    print("\n=== Results (same N) ===")
    print(f"[baseline] I = {Ib:.6e} ± {dIb:.2e} | sigma_hat = {sb:.6e} ± {dsb:.2e}")
    print(f"[NF costh] I = {In:.6e} ± {dIn:.2e} | sigma_hat = {sn:.6e} ± {dsn:.2e}")

    vr = (dIb / dIn) ** 2
    print(f"[improvement] variance reduction factor ≈ {vr:.2f}×")
    print(f"[timing] baseline integration: {t1 - t0:.2f}s")
    print(f"[timing] NF integration:       {t2 - t1:.2f}s")

    save_comparison(
        results_path,
        baseline={
            "I": float(Ib),
            "dI": float(dIb),
            "sigma_hat": float(sb),
            "dsigma_hat": float(dsb),
            "N": int(n_eval),
            "time_sec": float(t1 - t0),
        },
        nf={
            "I": float(In),
            "dI": float(dIn),
            "sigma_hat": float(sn),
            "dsigma_hat": float(dsn),
            "N": int(n_eval),
            "time_sec": float(t2 - t1),
        },
        meta={
            "Ecm": float(Ecm),
            "mZ": float(mZ),
            "steps": int(steps),
            "batch_size": int(batch_size),
            "hidden": int(hidden),
            "n_blocks": int(n_blocks),
            "seed": 0,
        },
    )

    # 3) Plots
    plot_training_curve(losses, plot_dir)
    plot_histograms(flow, Ecm, mZ, plot_dir, n_target=120000, n_nf=120000, seed=3)

    print(f"\n[done] plots saved to: {plot_dir.resolve()}")
    print("[done] files: training_loss.png, hist_costh_target_vs_nf.png, hist_phi_target_vs_nf.png")
    print(f"[done] results saved to: {results_path.resolve()}")


if __name__ == "__main__":
    main()