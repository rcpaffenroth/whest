"""Width-scaling of the confirmed equivariant winner (residual/mean, pure-W).

The question the whole thread built toward: does R² hold as width scales toward the
competition regime (256), where the flat model collapsed and the analytic route
dominates? The equivariant model's params are O(c²), width-independent (7.4k at
c=32 for ALL widths), so it *should* hold — this tests it.

Depth fixed at 8 (isolate width). N shrinks with width to bound RAM (storing W is
N·d²): 32→400k, 64→200k, 128→50k, 256→12k. So the width and data axes are partly
confounded at large width (esp. 256, ratio ~1.7 examples/param) — flagged. Prints
each row as it finishes (partial results survive a stop).
"""

import time

import torch
import torch.nn as nn

import equivariant as E   # EquivariantNet, dev; import does NOT run its sweep
import whest_toy as toy

dev = E.dev
DEPTH, M, C = 8, 8192, 32
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 1024, 120, 20
N_BY_W = {32: 400_000, 64: 200_000, 128: 50_000, 256: 12_000}
CHUNK_BY_W = {32: 2000, 64: 1024, 128: 512, 256: 256}   # bound the M·d activation on the 24GB GPU
WIDTHS = [64, 128, 256]                                  # 32 already banked at R²=0.983


def gen(seeds, d, chunk):
    """W (n,DEPTH,d,d) and MC mean F (n,d) — pure W, no UT."""
    Ws, Ff = [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s), k=DEPTH, d=d).float()
                         for s in seeds[s0:s0 + chunk]]).to(dev)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(1_000_000 + s0)
        z = torch.randn(M, d, generator=g, device=dev).unsqueeze(0).expand(B, M, d)
        for i in range(DEPTH):
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
    npar = sum(p.numel() for p in net.parameters())
    return 1 - best_te / const, npar


print(f"{'d':>4} {'N':>8} {'params':>7} {'R2(log)':>8} {'minutes':>8}   (flat LSTM @w32 ~0.30)", flush=True)
for d in WIDTHS:
    t0 = time.time()
    N = N_BY_W[d]; ch = CHUNK_BY_W[d]
    Wtr, Ftr = gen(list(range(N)), d, ch)
    Wva, Fva = gen(list(range(905_000, 910_000)), d, ch)
    Wte, Fte = gen(list(range(900_000, 905_000)), d, ch)
    torch.cuda.empty_cache()
    r2, npar = train(Wtr, Ftr, Wva, Fva, Wte, Fte)
    print(f"{d:>4} {N:>8} {npar:>7} {r2:>8.3f} {(time.time()-t0)/60:>8.1f}", flush=True)
    del Wtr, Ftr, Wva, Fva, Wte, Fte
    torch.cuda.empty_cache()
