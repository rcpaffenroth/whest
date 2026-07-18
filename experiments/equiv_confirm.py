"""Confirm the equivariant sweep winner over seeds (RESEARCH_LOG §10.1).

The single-seed sweep gave residual/mean2 0.982, gru/mean 0.977, gru/mean2 0.831 —
so both "residual vs gru" and "mean vs mean2 pool" need error bars. Run the 2x2
{residual,gru} x {mean,mean2} over 3 seeds at width 32, N=400k, pure-W. Reuses the
verified model from equivariant.py (imported, not re-implemented).
"""

import numpy as np
import torch
import torch.nn as nn

import equivariant as E   # EquivariantNet, gen, WIDTH, DEPTH, M, dev — import does NOT run its sweep

dev = E.dev
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 4096, 150, 25
SEEDS = [0, 1, 2]
CONTENDERS = [("residual", "mean"), ("residual", "mean2"), ("gru", "mean"), ("gru", "mean2")]

print(f"generating width-{E.WIDTH} data (once, reused across all runs) ...", flush=True)
Wtr, _, _, Ftr = E.gen(list(range(E.N_TRAIN)))
Wva, _, _, Fva = E.gen(list(range(905_000, 910_000)))
Wte, _, _, Fte = E.gen(list(range(900_000, 905_000)))
wstd = Wtr.std().item()
ylog = torch.log1p(Ftr); ym, ys = ylog.mean().item(), ylog.std().item()
const = (torch.log1p(Fte) - ym).pow(2).mean().item()
print(f"train {len(Wtr)}  test {len(Wte)}\n")


def train(phi, pool, seed):
    torch.manual_seed(seed)
    net = E.EquivariantNet(phi=phi, pool=pool, use_ut=False).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()

    def predict(W):
        out = []
        for b in range(0, len(W), BATCH):
            out.append(net((W[b:b + BATCH] / wstd).to(dev), None).cpu())
        return torch.cat(out)

    n = len(Wtr); best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            wb = (Wtr[idx] / wstd).to(dev)
            yb = ((torch.log1p(Ftr[idx]) - ym) / ys).to(dev)
            opt.zero_grad(); lf(net(wb, None), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            va = ((predict(Wva) * ys + ym - torch.log1p(Fva)) ** 2).mean().item()
            te = ((predict(Wte) * ys + ym - torch.log1p(Fte)) ** 2).mean().item()
        if va < best_va: best_va, best_te, wait = va, te, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    return 1 - best_te / const


print(f"{'phi':9s} {'pool':6s} {'R2 mean':>8s} {'std':>6s}   seeds")
for phi, pool in CONTENDERS:
    r2s = [train(phi, pool, s) for s in SEEDS]
    print(f"{phi:9s} {pool:6s} {np.mean(r2s):>8.3f} {np.std(r2s):>6.3f}   "
          f"{[round(r, 3) for r in r2s]}", flush=True)
