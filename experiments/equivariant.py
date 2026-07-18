"""Permutation-equivariant matrix-state model for F(W)  (RESEARCH_LOG §10.1/10.2).

State  H ∈ R^{B×d×c}: a d×c matrix per network that co-transforms with the
activation vector (H ↦ P H under the layer's coordinate permutation). Its c
columns are learned probe-vectors living in the layer's coordinate space.

Transition at layer l uses the REAL weight matrix as the operator (no flatten):
    Hp = W_l H        (mean message,  W_l H)
    H2 = W_l∘W_l H    (variance message, entrywise-squared weights)
then a Deep-Sets pool over rows (the mean-field order parameters, §10.2) is
appended and a SHARED row-cell Φ produces H_l. Readout F̂ = H_K w.

Equivariant by construction (permutation matrices are orthogonal): W_l H ↦
P_l(W_l H); row-cells and column-means commute with P. Learned params are O(c²),
INDEPENDENT of width d — the point of the whole exercise (beats the flat LSTM's
d²-token wall, which stalled at R²≈0.27-0.30 at width 32).

Swappable knobs (we sweep them): Φ ∈ {gru,lstm,residual}, pool ∈ {mean,mean2,attn},
use_ut ∈ {False,True} (the hybrid G(W,UT): stack per-coordinate UT moments in).
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn

import whest_toy as toy

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
WIDTH, DEPTH = 32, 8
M = 8192
N_TRAIN = 400_000
C = 32                                  # channels per row
LR, BATCH, EPOCHS, PATIENCE = 1e-3, 4096, 150, 25


# ------------------------------- the model ----------------------------------
class RowAttn(nn.Module):
    """Single-head self-attention over the d rows (permutation-equivariant pool)."""
    def __init__(s, dim):
        super().__init__(); s.q = nn.Linear(dim, dim); s.k = nn.Linear(dim, dim)
        s.v = nn.Linear(dim, dim); s.scale = dim ** -0.5
    def forward(s, X):                                   # X: (B,d,dim)
        A = torch.softmax(s.q(X) @ s.k(X).transpose(1, 2) * s.scale, dim=-1)
        return A @ s.v(X)


class EquivariantNet(nn.Module):
    def __init__(s, c=C, phi="gru", pool="mean2", use_ut=False):
        super().__init__()
        s.c, s.phi, s.pool, s.use_ut = c, phi, pool, use_ut
        base = 2 * c                                     # X = [Hp, H2]
        if pool == "mean":   pool_extra = 2 * c
        elif pool == "mean2": pool_extra = 4 * c
        elif pool == "attn": pool_extra = 2 * c; s.attn = RowAttn(2 * c)
        feat = base + pool_extra + (2 if use_ut else 0)  # +2: per-row UT (mean,var)
        s.h0 = nn.Parameter(torch.zeros(1, 1, c))        # symmetric input state (all rows equal)
        if phi == "gru":       s.cell = nn.GRUCell(feat, c)
        elif phi == "lstm":    s.cell = nn.LSTMCell(feat, c); s.c0 = nn.Parameter(torch.zeros(1, 1, c))
        elif phi == "residual": s.mlp = nn.Sequential(nn.Linear(feat, c), nn.GELU(), nn.Linear(c, c))
        s.norm = nn.LayerNorm(c)                          # per-row, equivariant
        s.head = nn.Linear(c, 1)

    def forward(s, W, UT=None):                           # W:(B,K,d,d)  UT:(B,K,d,2) or None
        B, K, d, _ = W.shape
        H = s.h0.expand(B, d, s.c)
        if s.phi == "lstm": Cst = s.c0.expand(B, d, s.c)
        for l in range(K):
            Wl = W[:, l]                                  # (B,d,d)
            Hp = torch.einsum("bjk,bkc->bjc", Wl, H)      # W_l H
            H2 = torch.einsum("bjk,bkc->bjc", Wl * Wl, H) # W_l∘2 H
            X = torch.cat([Hp, H2], -1)                   # (B,d,2c)
            if s.pool == "mean":
                g = [X.mean(1, keepdim=True).expand(-1, d, -1)]
            elif s.pool == "mean2":
                g = [X.mean(1, keepdim=True).expand(-1, d, -1),
                     (X * X).mean(1, keepdim=True).expand(-1, d, -1)]
            elif s.pool == "attn":
                g = [s.attn(X)]
            feat = [X] + g + ([UT[:, l]] if s.use_ut else [])
            f = torch.cat(feat, -1).reshape(B * d, -1)
            if s.phi == "gru":
                H = s.cell(f, Hp.reshape(B * d, -1)).view(B, d, s.c)
            elif s.phi == "lstm":
                Cp = torch.einsum("bjk,bkc->bjc", Wl, Cst)      # propagate cell state too
                h, cc = s.cell(f, (Hp.reshape(B * d, -1), Cp.reshape(B * d, -1)))
                H = h.view(B, d, s.c); Cst = cc.view(B, d, s.c)
            elif s.phi == "residual":
                H = Hp + s.mlp(f).view(B, d, s.c)
            H = s.norm(H)
        return s.head(H).squeeze(-1)                      # (B,d)


# ------------------------------- data ---------------------------------------
def gen(seeds, chunk=4000, input_seed=1_000_000):
    """W (n,K,d,d), UT per-layer mean/var (n,K,d), MC mean F (n,d)."""
    d = WIDTH
    S = (torch.cat([torch.eye(d), -torch.eye(d)], 0) * d ** 0.5).to(dev)   # fixed sigma set
    Ws, Um, Uv, Ff = [], [], [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s), k=DEPTH, d=d).float()
                         for s in seeds[s0:s0 + chunk]]).to(dev)           # (B,K,d,d)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        z = torch.randn(M, d, generator=g, device=dev).unsqueeze(0).expand(B, M, d)
        for i in range(DEPTH):
            z = torch.relu(torch.einsum("bpe,bfe->bpf", z, W[:, i]))
        Ff.append(z.mean(1).cpu())
        u = S.unsqueeze(0).expand(B, 2 * d, d); mns, vrs = [], []
        for i in range(DEPTH):
            u = torch.relu(torch.einsum("bpe,bfe->bpf", u, W[:, i]))
            mns.append(u.mean(1)); vrs.append(u.var(1))
        Ws.append(W.cpu()); Um.append(torch.stack(mns, 1).cpu()); Uv.append(torch.stack(vrs, 1).cpu())
    return torch.cat(Ws), torch.cat(Um), torch.cat(Uv), torch.cat(Ff)


def ut_of(Um, Uv): return torch.stack([Um, Uv], -1)      # (n,K,d,2)


def run(phi, pool, use_ut):
    torch.manual_seed(0)
    net = EquivariantNet(phi=phi, pool=pool, use_ut=use_ut).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()
    UTtr, UTva, UTte = (ut_of(Umtr, Uvtr), ut_of(Umva, Uvva), ut_of(Umte, Uvte)) if use_ut else (None, None, None)

    def predict(W, UT):                                  # batched eval, streamed to GPU
        out = []
        for b in range(0, len(W), BATCH):
            wb = (W[b:b + BATCH] / wstd).to(dev)
            ub = UT[b:b + BATCH].to(dev) if use_ut else None
            out.append(net(wb, ub).cpu())
        return torch.cat(out)

    n = len(Wtr); best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            wb = (Wtr[idx] / wstd).to(dev)
            ub = UTtr[idx].to(dev) if use_ut else None
            yb = ((torch.log1p(Ftr[idx]) - ym) / ys).to(dev)
            opt.zero_grad()
            loss = lf(net(wb, ub), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            va = ((predict(Wva, UTva) * ys + ym - torch.log1p(Fva)) ** 2).mean().item()
            te = ((predict(Wte, UTte) * ys + ym - torch.log1p(Fte)) ** 2).mean().item()
        if va < best_va: best_va, best_te, wait = va, te, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    npar = sum(p.numel() for p in net.parameters())
    return 1 - best_te / const, npar


CONFIGS = [
    ("gru", "mean2", False), ("lstm", "mean2", False), ("residual", "mean2", False),   # Φ sweep
    ("gru", "mean", False), ("gru", "attn", False),                                     # pool sweep
    ("gru", "mean2", True),                                                             # +UT hybrid
]


def main():
    global Wtr, Umtr, Uvtr, Ftr, Wva, Umva, Uvva, Fva, Wte, Umte, Uvte, Fte, wstd, ym, ys, const
    print(f"generating width-{WIDTH} data ...", flush=True)
    Wtr, Umtr, Uvtr, Ftr = gen(list(range(N_TRAIN)))
    Wva, Umva, Uvva, Fva = gen(list(range(905_000, 910_000)))
    Wte, Umte, Uvte, Fte = gen(list(range(900_000, 905_000)))
    wstd = Wtr.std().item()                               # scalar (keeps equivariance)
    ylog = torch.log1p(Ftr); ym, ys = ylog.mean().item(), ylog.std().item()
    const = (torch.log1p(Fte) - ym).pow(2).mean().item() # = Var(log1p F_te)
    print(f"train {N_TRAIN}  test {len(Fte)}  wstd {wstd:.3f}  logmean {ym:.3f} logstd {ys:.3f}\n")
    print(f"{'phi':9s} {'pool':6s} {'input':5s} {'params':>8s} {'R2(log)':>8s}   (flat LSTM @w32 ~0.27-0.30)")
    for phi, pool, use_ut in CONFIGS:
        r2, npar = run(phi, pool, use_ut)
        print(f"{phi:9s} {pool:6s} {'W+UT' if use_ut else 'W':5s} {npar:>8d} {r2:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
