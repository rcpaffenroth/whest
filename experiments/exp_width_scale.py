"""(a) Width-scaling study: does learning-from-W survive the approach to the RMT limit?

The competition is width 256; our toy is width 8, far from the random-matrix /
mean-field limit. Width is not just a cost knob -- it changes the NATURE of the
target. Mean-field theory (RESEARCH_LOG §8.1) says that as width grows, every
neuron's mean concentrates on a UNIVERSAL constant (0.564 at He criticality) and
the realization-specific signal shrinks as O(1/width). So the question is whether
the recurrent, learn-from-W approach keeps working as the signal concentrates, or
the problem collapses into tiny, analytic-friendly fluctuations.

We fix DEPTH=8 (every width is alive there) and sweep width d in {8,16,32}, the
ONLY variable. For each width we report:
  concentration -- mean of F (-> 0.564?), Var(F) and CV (-> 0 as ~1/width?), tail;
  learnability  -- held-out R^2 (log-space) of an LSTM reading per-layer W tokens,
                   and of the same LSTM reading per-layer UT moment features.

Data is kept on CPU and streamed to the GPU per batch (flat weights at width 32
are several GB).
"""

import torch
import torch.nn as nn

import whest_toy as toy

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH = 8
WIDTHS = [8, 16, 32]
N = 150_000
M = 8192
HIDDEN, LAYERS, LR, BATCH, EPOCHS, PATIENCE = 256, 2, 1e-3, 4096, 200, 30


def gen(seeds, d, chunk=4000, input_seed=1_000_000):
    """Per network at width d, depth DEPTH: flat(W), per-layer UT (mean, cov-diag), MC mean F."""
    S = (torch.cat([torch.eye(d), -torch.eye(d)], 0) * (d ** 0.5)).to(dev)     # (2d, d) fixed sigma set
    Wf, Um, Uc, Ff = [], [], [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s), k=DEPTH, d=d).float()
                         for s in seeds[s0:s0 + chunk]]).to(dev)               # (B,DEPTH,d,d)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        z = torch.randn(M, d, generator=g, device=dev).unsqueeze(0).expand(B, M, d)
        for i in range(DEPTH):
            z = torch.relu(torch.einsum("bpe,bfe->bpf", z, W[:, i]))
        Ff.append(z.mean(1).cpu())                                            # (B,d) MC truth
        u = S.unsqueeze(0).expand(B, 2 * d, d)
        means, covs = [], []
        for i in range(DEPTH):
            u = torch.relu(torch.einsum("bpe,bfe->bpf", u, W[:, i]))
            means.append(u.mean(1)); covs.append(u.var(1))
        Wf.append(W.reshape(B, -1).cpu())
        Um.append(torch.stack(means, 1).cpu()); Uc.append(torch.stack(covs, 1).cpu())
    return torch.cat(Wf), torch.cat(Um), torch.cat(Uc), torch.cat(Ff)


class LSTMNet(nn.Module):
    def __init__(self, in_feat, d_out):
        super().__init__()
        self.rnn = nn.LSTM(in_feat, HIDDEN, LAYERS, batch_first=True); self.head = nn.Linear(HIDDEN, d_out)
    def forward(self, x): return self.head(self.rnn(x)[0][:, -1])


def fit(Xtr, Ytr, Xva, Yva, Xte, Yte):
    """Xtr/Xva/Xte: (n,DEPTH,feat) on CPU. Target log1p(F), standardized. R^2 in log-space."""
    torch.manual_seed(0)
    xm, xs = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8
    yt = torch.log1p(Ytr); ym, ys = yt.mean(0), yt.std(0) + 1e-8
    zx = lambda X: (X - xm) / xs                                              # stays on CPU
    Xtr_d = zx(Xtr); Ytr_d = ((yt - ym) / ys)
    Xva_d, Yva_t = zx(Xva).to(dev), torch.log1p(Yva)
    Xte_d, Yte_t = zx(Xte).to(dev), torch.log1p(Yte)
    net = LSTMNet(Xtr.shape[-1], Ytr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()
    unz = lambda P: P.cpu() * ys + ym
    n = len(Xtr_d); best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            lf(net(Xtr_d[idx].to(dev)), Ytr_d[idx].to(dev)).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            v = ((unz(net(Xva_d)) - Yva_t) ** 2).mean().item()
            t = ((unz(net(Xte_d)) - Yte_t) ** 2).mean().item()
        if v < best_va: best_va, best_te, wait = v, t, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    const = ((yt.mean(0) - Yte_t) ** 2).mean().item()
    return 1 - best_te / const


print(f"{'d':>3} {'token':>6} {'meanF':>7} {'Var(F)':>9} {'CV':>6} {'p99/med':>8} "
      f"{'R2(W)':>7} {'R2(UT)':>7}")
for d in WIDTHS:
    Wf, Um, Uc, Ff = gen(list(range(N)), d)
    va = gen(list(range(905_000, 910_000)), d)
    te = gen(list(range(900_000, 905_000)), d)

    mu, var = Ff.mean().item(), Ff.var().item()
    cv = var ** 0.5 / mu
    nrm = Ff.norm(dim=1)
    tail = (torch.quantile(nrm, 0.99) / nrm.median()).item()

    Wseq = lambda W: W.reshape(-1, DEPTH, d * d)
    UTseq = lambda Um, Uc: torch.cat([Um, Uc], dim=-1)
    r2_W = fit(Wseq(Wf), Ff, Wseq(va[0]), va[3], Wseq(te[0]), te[3])
    r2_UT = fit(UTseq(Um, Uc), Ff, UTseq(va[1], va[2]), va[3], UTseq(te[1], te[2]), te[3])
    print(f"{d:>3} {d*d:>6} {mu:>7.3f} {var:>9.2e} {cv:>6.2f} {tail:>8.2f} "
          f"{r2_W:>7.3f} {r2_UT:>7.3f}", flush=True)
