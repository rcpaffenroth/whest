"""Corrector v1: UT propagation + EXACT analytic rectification of the final layer.

UT carries the true (non-Gaussian) joint through the first depth-1 layers -- it is
the best moment propagator we have. The probe showed the FINAL pre-activation is
Gaussian given correct penultimate moments, so instead of UT's noisy final-layer
average  (1/P) sum ReLU(h_j),  use the exact scalar rectification
    mu_j = sigma_j * R(alpha_j),   alpha_j = muh_j / sigma_j,   R(a)=a*Phi(a)+phi(a),
with (muh, sigma) read off the penultimate UT cloud:
    muh   = W_last^T mu_prev
    sigma_j^2 = W_last[:,j]^T Sigma_prev W_last[:,j]     (full)   or  sum_a W[a,j]^2 var_a (diag)

Compare direct-UT vs analytic-last (full / diag) over several MLPs x rotation seeds,
so sampling variance shows up as spread. Lower final-layer mean MSE is better.
"""
import numpy as np
from scipy.stats import norm
import whestbench as wb

WIDTH, DEPTH = 256, 32
SEEDS = [0, 1, 2]
ROT_SEEDS = range(8)
ROTATIONS = 12
N_GT = 500_000

Phi, phi = norm.cdf, norm.pdf
R = lambda a: a * Phi(a) + phi(a)


def ut_cloud(W, rotations, rng):
    """Propagate the randomized-UT sigma points through layers W; return the cloud."""
    r = np.sqrt(WIDTH); blocks = []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
        blocks += [r * Q.T, -r * Q.T]
    h = np.concatenate(blocks, 0)
    for w in W:
        h = np.maximum(h @ w, 0.0)
    return h


def estimates(W, rotations, rng):
    cloud = ut_cloud(W[:-1], rotations, rng)          # penultimate activations
    Wl = W[-1]
    direct = np.maximum(cloud @ Wl, 0.0).mean(0)      # UT's usual final-layer average

    mu_prev = cloud.mean(0)
    muh = Wl.T @ mu_prev
    # full covariance path
    Sig = np.cov(cloud, rowvar=False, bias=True)
    var_full = np.einsum('ac,bc,ab->c', Wl, Wl, Sig)
    s = np.sqrt(np.maximum(var_full, 1e-12))
    analytic_full = s * R(muh / s)
    # diagonal path (no full cov)
    var_prev = cloud.var(0)
    var_diag = (Wl * Wl).T @ var_prev
    sd = np.sqrt(np.maximum(var_diag, 1e-12))
    analytic_diag = sd * R(muh / sd)
    return direct, analytic_full, analytic_diag


print(f"whestbench {WIDTH}x{DEPTH}  rotations={ROTATIONS} ({2*WIDTH*ROTATIONS} pts)  "
      f"GT N={N_GT:,}  {len(list(ROT_SEEDS))} rotation seeds\n")
print(f"{'seed':>4} {'direct(UT)':>22} {'analytic_full':>22} {'analytic_diag':>22}   (mean +/- std of MSE)")
tot = {"direct": [], "full": [], "diag": []}
for seed in SEEDS:
    mlp = wb.sample_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    truth = np.asarray(wb.sample_layer_statistics(mlp, N_GT)[0][-1], dtype=np.float64)
    d, f, g = [], [], []
    for rs in ROT_SEEDS:
        rng = np.random.default_rng(1000 * seed + rs)
        de, fu, di = estimates(W, ROTATIONS, rng)
        d.append(((de - truth) ** 2).mean())
        f.append(((fu - truth) ** 2).mean())
        g.append(((di - truth) ** 2).mean())
    tot["direct"] += d; tot["full"] += f; tot["diag"] += g
    print(f"{seed:>4} {np.mean(d):>11.3e}+-{np.std(d):.1e} {np.mean(f):>11.3e}+-{np.std(f):.1e} "
          f"{np.mean(g):>11.3e}+-{np.std(g):.1e}")

print("\npooled mean over all MLPs x seeds:")
for k in tot:
    print(f"  {k:8s} {np.mean(tot[k]):.3e}  (median {np.median(tot[k]):.3e})")
