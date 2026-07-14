"""Probe: is the per-neuron final-layer mean explained by basis-aligned column
statistics of the last weight matrix?

For output neuron j of the final layer, the exact answer is
    E[z_{L,j}] = E[ReLU(h_j)],   h_j = <W[:,j], z_{prev}>,
so h_j has mean  mu_j = <W[:,j], mu_prev>  and variance  s_j^2 = W[:,j]^T Sigma_prev W[:,j].
If h_j were Gaussian, E[ReLU(h_j)] = mu_j Phi(mu_j/s_j) + s_j phi(mu_j/s_j)  (= cov_prop).

We test, across neurons j and across several real whestbench MLPs:
  (A) how well the TWO raw weight stats  align_j=<W[:,j],mu_prev>, colnorm_j=||W[:,j]||^2
      explain the per-neuron truth deviations;
  (B) how well the exact Gaussian formula (cov_prop) explains the truth, and what its
      residual (the NON-Gaussian part) looks like -- and whether the raw stats explain THAT;
  (C) the UT residual (truth - UT) and its correlation with the same stats.
"""
import numpy as np
from scipy.stats import norm
import whestbench as wb

WIDTH, DEPTH = 256, 32
SEEDS = [0, 1, 2, 3, 4, 5]
N_GT = 1_000_000          # ground-truth MC samples
CHUNK = 25_000
UT_ROTATIONS = 12         # ~ our submission's budget


def mc_stats(mlp, n, chunk, rng):
    """Chunked MC: return truth mean of final layer, mean & covariance of prev layer."""
    W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    sum_prev = np.zeros(WIDTH); gram_prev = np.zeros((WIDTH, WIDTH)); sum_last = np.zeros(WIDTH)
    done = 0
    while done < n:
        b = min(chunk, n - done)
        h = rng.standard_normal((b, WIDTH))
        for w in W[:-1]:
            h = np.maximum(h @ w, 0.0)
        z_prev = h                                   # activations feeding the last layer
        z_last = np.maximum(z_prev @ W[-1], 0.0)
        sum_prev += z_prev.sum(0); gram_prev += z_prev.T @ z_prev; sum_last += z_last.sum(0)
        done += b
    mu_prev = sum_prev / n
    Sigma_prev = gram_prev / n - np.outer(mu_prev, mu_prev)
    truth = sum_last / n
    return truth, mu_prev, Sigma_prev


def ut_final(mlp, rotations, rng):
    W = [np.asarray(w, dtype=np.float64) for w in mlp.weights]
    r = np.sqrt(WIDTH); blocks = []
    for _ in range(rotations):
        Q, _ = np.linalg.qr(rng.standard_normal((WIDTH, WIDTH)))
        blocks += [r * Q.T, -r * Q.T]
    h = np.concatenate(blocks, 0)
    for w in W:
        h = np.maximum(h @ w, 0.0)
    return h.mean(0)


def r2(y, X):
    """R^2 of OLS regressing y on columns of X plus intercept."""
    A = np.column_stack([np.ones_like(y), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - resid.var() / y.var()


print(f"whestbench MLPs width={WIDTH} depth={DEPTH}  GT N={N_GT:,}  UT rot={UT_ROTATIONS}\n")
agg = {k: [] for k in ["std_truth", "cA", "cN", "r2_AN", "r2_gauss", "r2_res_AN", "r2_ut_AN", "ut_bias_frac"]}
for seed in SEEDS:
    mlp = wb.sample_mlp(width=WIDTH, depth=DEPTH, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    truth, mu_prev, Sigma_prev = mc_stats(mlp, N_GT, CHUNK, rng)
    Wlast = np.asarray(mlp.weights[-1], dtype=np.float64)

    align = Wlast.T @ mu_prev                                  # <W[:,j], mu_prev>  (pre-activation mean)
    colnorm = (Wlast**2).sum(0)                                # ||W[:,j]||^2
    s = np.sqrt(np.einsum('ac,bc,ab->c', Wlast, Wlast, Sigma_prev))  # exact pre-activation std s_j^2 = W[:,j]^T Sigma W[:,j]
    gauss = align * norm.cdf(align / s) + s * norm.pdf(align / s)    # cov_prop per-neuron

    ut = ut_final(mlp, UT_ROTATIONS, rng)

    dtruth = truth - truth.mean()                              # per-neuron fluctuation about the constant
    cA = np.corrcoef(dtruth, align)[0, 1]
    cN = np.corrcoef(dtruth, colnorm)[0, 1]
    r2_AN = r2(truth, np.column_stack([align, colnorm]))       # raw stats -> truth
    r2_gauss = 1.0 - ((truth - gauss)**2).mean() / truth.var() # cov_prop -> truth (as R^2-like)
    resid_g = truth - gauss                                    # non-Gaussian residual
    r2_res_AN = r2(resid_g, np.column_stack([align, colnorm, s]))
    ut_res = truth - ut
    r2_ut_AN = r2(ut_res, np.column_stack([align, colnorm, s]))
    # crude bias/variance split of UT residual: compare to a second independent UT draw
    ut2 = ut_final(mlp, UT_ROTATIONS, np.random.default_rng(9000 + seed))
    var_ut = ((ut - ut2)**2).mean() / 2                        # ~ sampling variance of one UT draw
    mse_ut = (ut_res**2).mean()
    ut_bias_frac = max(0.0, 1.0 - var_ut / mse_ut)             # fraction of UT MSE that is bias^2

    for k, v in dict(std_truth=dtruth.std(), cA=cA, cN=cN, r2_AN=r2_AN, r2_gauss=r2_gauss,
                     r2_res_AN=r2_res_AN, r2_ut_AN=r2_ut_AN, ut_bias_frac=ut_bias_frac).items():
        agg[k].append(v)
    print(f"seed {seed}: std(truth)={dtruth.std():.4f} | corr align={cA:+.3f} colnorm={cN:+.3f} "
          f"| R2(truth|A,N)={r2_AN:.3f} R2(truth|cov_prop)={r2_gauss:.3f} "
          f"| R2(nonGauss resid|A,N,s)={r2_res_AN:.3f} | R2(UT resid|A,N,s)={r2_ut_AN:.3f} "
          f"| UT bias-frac={ut_bias_frac:.2f}")

print("\nmeans over MLPs:")
for k in agg:
    print(f"  {k:14s} {np.mean(agg[k]):+.4f}")
