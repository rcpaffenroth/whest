"""Corrector v3: two-radius randomized UT -- kill UT's radial (degree-4) bias.

Spherical-radial view: E_{x~N(0,I_n)}[f] = E_rho E_angular[f], rho^2 = ||x||^2 ~ chi^2_n.
The base UT puts ALL mass at one radius rho=sqrt(n): it matches E[rho^2]=n but NOT
E[rho^4]=n(n+2) -- the degree-4 radial bias. Random rotations already unbias the
angular part, so this radial mismatch is UT's ONLY deterministic bias.

Fix: a 2-point Gauss quadrature on the radial measure (t=rho^2 ~ chi^2_n), matching
moments E[t^0..3]. Two radii rho_1, rho_2 with weights w_1, w_2, each carried by the
+-e_i stencil under random rotations. Same point budget as base UT for a fair test.
"""
import numpy as np
import whestbench as wb

WIDTH, DEPTH = 256, 32
SEEDS = list(range(8))
ROT_SEEDS = range(8)
N_GT = 500_000
n = WIDTH


def radial_2point(n):
    """2-point Gauss nodes/weights for t ~ chi^2_n (matches E[t^0..3])."""
    mu0, mu1 = 1.0, float(n)
    mu2 = n * (n + 2.0)
    mu3 = n * (n + 2.0) * (n + 4.0)
    # orthogonal poly pi2(t)=t^2 - a t - b, orthogonal to {1, t}:
    A = np.array([[mu1, mu0], [mu2, mu1]]); rhs = np.array([mu2, mu3])
    a, b = np.linalg.solve(A, rhs)
    t = np.roots([1.0, -a, -b])                 # two nodes
    V = np.array([[1.0, 1.0], [t[0], t[1]]])
    w = np.linalg.solve(V, np.array([mu0, mu1]))  # weights
    return np.sqrt(t), w


def ut_means(W, rotations, rng):
    """Base UT (single radius sqrt(n)), equal weights."""
    r = np.sqrt(n); blocks = []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        blocks += [r * Q.T, -r * Q.T]
    h = np.concatenate(blocks, 0)
    out = []
    for w in W:
        h = np.maximum(h @ w, 0.0); out.append(h.mean(0))
    return out[-1]


def two_radius_means(W, rotations, radii, rw, rng):
    """Two-radius randomized UT with radial weights rw."""
    pts, wts = [], []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        axes = Q.T
        for rho, w_r in zip(radii, rw):
            pts += [rho * axes, -rho * axes]
            wts += [np.full(2 * n, w_r / (2 * n))]           # weight per point
    h = np.concatenate(pts, 0)
    wts = np.concatenate(wts) / rotations                    # sum(wts)=1
    for w in W:
        h = np.maximum(h @ w, 0.0)
    return wts @ h                                           # weighted final-layer mean


radii, rw = radial_2point(n)
print(f"whestbench {WIDTH}x{DEPTH}  radii={radii.round(3)} (sqrt(n)={np.sqrt(n):.3f}) "
      f"weights={rw.round(4)}  GT N={N_GT:,}\n")
# fair budget: base UT uses 2n*ROT_UT points; 2-radius uses 4n*ROT_2R points -> ROT_2R = ROT_UT/2
ROT_UT, ROT_2R = 12, 6
print(f"base UT: {ROT_UT} rot = {2*n*ROT_UT} pts   |   2-radius: {ROT_2R} rot = {4*n*ROT_2R} pts\n")
print(f"{'seed':>4} {'UT':>22} {'two-radius':>22}   (mean +/- std of MSE)")
tot_ut, tot_2r = [], []
for seed in SEEDS:
    mlp = wb.sample_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    truth = np.asarray(wb.sample_layer_statistics(mlp, N_GT)[0][-1], dtype=np.float64)
    u, t2 = [], []
    for rs in ROT_SEEDS:
        u.append(((ut_means(W, ROT_UT, np.random.default_rng(1000*seed+rs)) - truth)**2).mean())
        t2.append(((two_radius_means(W, ROT_2R, radii, rw, np.random.default_rng(7000+1000*seed+rs)) - truth)**2).mean())
    tot_ut += u; tot_2r += t2
    print(f"{seed:>4} {np.mean(u):>11.3e}+-{np.std(u):.1e} {np.mean(t2):>11.3e}+-{np.std(t2):.1e}  ({np.mean(u)/np.mean(t2):.2f}x)")

print(f"\npooled: UT {np.mean(tot_ut):.3e}  |  two-radius {np.mean(tot_2r):.3e}  "
      f"|  {np.mean(tot_ut)/np.mean(tot_2r):.2f}x")
