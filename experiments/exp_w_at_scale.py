"""(b) Does raw-W learning close the gap to UT with more DATA?

The width study (a) showed R^2(W) falling with width (0.46->0.27 at 150k) while
R^2(UT) sat at ~0.998 -- but at FIXED data, so R^2(W) is confounded with data-
starvation (bigger width = bigger token = more data needed). This test scales the
DATA for the raw-W LSTM at two widths and asks: does R^2(W) climb toward UT's
~1.0, or plateau well below it?

  width 8  (UT baseline R^2~0.995): N in {300k, 600k, 1.2M}
  width 32 (UT baseline R^2~0.998): N in {150k, 400k}

Same LSTM as everywhere (hidden 256, 2 layers), target log1p(F), leakage-proof
val/test, data on CPU streamed to GPU per batch.
"""

import torch
import torch.nn as nn

import whest_toy as toy

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH = 8
M = 8192
HIDDEN, LAYERS, LR, BATCH, EPOCHS, PATIENCE = 256, 2, 1e-3, 4096, 250, 30
WIDTHS_N = {8: [300_000, 600_000, 1_200_000], 32: [150_000, 400_000]}


def gen(seeds, d, chunk=4000, input_seed=1_000_000):
    """Per network at width d, depth DEPTH: flat(W) and MC mean F (no UT needed here)."""
    Wf, Ff = [], []
    for s0 in range(0, len(seeds), chunk):
        W = torch.stack([toy.sample_weights(int(s), k=DEPTH, d=d).float()
                         for s in seeds[s0:s0 + chunk]]).to(dev)
        B = W.shape[0]
        g = torch.Generator(device=dev).manual_seed(input_seed + s0)
        z = torch.randn(M, d, generator=g, device=dev).unsqueeze(0).expand(B, M, d)
        for i in range(DEPTH):
            z = torch.relu(torch.einsum("bpe,bfe->bpf", z, W[:, i]))
        Wf.append(W.reshape(B, -1).cpu()); Ff.append(z.mean(1).cpu())
    return torch.cat(Wf), torch.cat(Ff)


class LSTMNet(nn.Module):
    def __init__(self, in_feat, d_out):
        super().__init__()
        self.rnn = nn.LSTM(in_feat, HIDDEN, LAYERS, batch_first=True); self.head = nn.Linear(HIDDEN, d_out)
    def forward(self, x): return self.head(self.rnn(x)[0][:, -1])


def fit(Xtr, Ytr, Xva, Yva, Xte, Yte, d):
    torch.manual_seed(0)
    seq = lambda X: X.reshape(-1, DEPTH, d * d)
    Xtr, Xva, Xte = seq(Xtr), seq(Xva), seq(Xte)
    xm, xs = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-8
    zx = lambda X: (X - xm) / xs
    yt = torch.log1p(Ytr); ym, ys = yt.mean(0), yt.std(0) + 1e-8
    Xtr_d, Ytr_d = zx(Xtr), (yt - ym) / ys
    Xva_d, Yva_t = zx(Xva).to(dev), torch.log1p(Yva)
    Xte_d, Yte_t = zx(Xte).to(dev), torch.log1p(Yte)
    net = LSTMNet(d * d, Ytr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lf = nn.MSELoss()
    unz = lambda P: P.cpu() * ys + ym
    n = len(Xtr_d); best_va, best_te, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]; opt.zero_grad()
            lf(net(Xtr_d[idx].to(dev)), Ytr_d[idx].to(dev)).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            v = ((unz(net(Xva_d)) - Yva_t) ** 2).mean().item()
            t = ((unz(net(Xte_d)) - Yte_t) ** 2).mean().item()
        if v < best_va: best_va, best_te, wait = v, t, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    return 1 - best_te / ((yt.mean(0) - Yte_t) ** 2).mean().item()


print(f"{'d':>3} {'N':>9} {'R2(W)':>7}   (UT baseline: d8~0.995, d32~0.998)")
for d, Ns in WIDTHS_N.items():
    Xva, Yva = gen(list(range(905_000, 910_000)), d)
    Xte, Yte = gen(list(range(900_000, 905_000)), d)
    for N in Ns:
        Xtr, Ytr = gen(list(range(N)), d)
        r2 = fit(Xtr, Ytr, Xva, Yva, Xte, Yte, d)
        print(f"{d:>3} {N:>9} {r2:>7.3f}", flush=True)
        del Xtr, Ytr
