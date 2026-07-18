"""Depth-scaling of the equivariant winner (residual/mean, pure-W) toward competition depth.

Width study showed R² rises with width. Depth is the other, harder axis — the "time"
of the dynamical system, where inter-layer correlations and closure-drift compound
(the competition is depth 32). Fix WIDTH=64 (safely alive at depth 32) and sweep
depth in {8,16,24,32}. The equivariant model's params are independent of BOTH width
and depth (shared cell), so 5.3k params at every point.

N shrinks with depth to bound RAM (storing W is N·K·d²): 200k/100k/75k/50k. So depth
and data are partly confounded, but the model is very data-efficient (width study hit
0.993 at 12k), so the effect should be small — flagged. Prints each row as it finishes.
"""

import time

import torch
import torch.nn as nn

import equivariant as E
import whest_toy as toy

dev = E.dev
WIDTH, M, C = 64, 8192, 32
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 1024, 120, 20
N_BY_DEPTH = {8: 200_000, 16: 100_000, 24: 75_000, 32: 50_000}
CHUNK = 1024


def gen(seeds, K, chunk=CHUNK):
    """W (n,K,WIDTH,WIDTH) and MC mean F (n,WIDTH) — pure W."""
    d = WIDTH
    Ws, Ff = [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s), k=K, d=d).float()
                         for s in seeds[s0:s0 + chunk]]).to(dev)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(1_000_000 + s0)
        z = torch.randn(M, d, generator=g, device=dev).unsqueeze(0).expand(B, M, d)
        for i in range(K):
            z = torch.relu(torch.einsum("bpe,bfe->bpf", z, W[:, i]))
        Ws.append(W.cpu()); Ff.append(z.mean(1).cpu())
        del W, z
        torch.cuda.empty_cache()
    return torch.cat(Ws), torch.cat(Ff)


def train(Wtr, Ftr, Wva, Fva, Wte, Fte):
    torch.manual_seed(0)
    net = E.EquivariantNet(c=C, phi="residual", pool="mean", use_ut=False).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()
    wstd = Wtr.std().item()
    ylog = torch.log1p(Ftr); ym, ys = ylog.mean().item(), ylog.std().item()
    const = (torch.log1p(Fte) - ym).pow(2).mean().item()

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
    return 1 - best_te / const, sum(p.numel() for p in net.parameters())


print(f"width {WIDTH}  |  {'K':>3} {'N':>8} {'params':>7} {'R2(log)':>8} {'minutes':>8}", flush=True)
for K in [8, 16, 24, 32]:
    t0 = time.time()
    N = N_BY_DEPTH[K]
    Wtr, Ftr = gen(list(range(N)), K)
    Wva, Fva = gen(list(range(905_000, 910_000)), K)
    Wte, Fte = gen(list(range(900_000, 905_000)), K)
    torch.cuda.empty_cache()
    r2, npar = train(Wtr, Ftr, Wva, Fva, Wte, Fte)
    print(f"            {K:>3} {N:>8} {npar:>7} {r2:>8.3f} {(time.time()-t0)/60:>8.1f}", flush=True)
    del Wtr, Ftr, Wva, Fva, Wte, Fte
    torch.cuda.empty_cache()
