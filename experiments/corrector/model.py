"""model.py -- the one recurrent corrector (PLAN_v1 section 2).

    h_k = tanh( g([pooled_k, h_{k-1}]) ),   h_0 = 0        # Channel 3: global recurrence
    correction_j = out_scale * R([neuron_j, h_L])          # per-neuron readout, zero-init
    estimate_j   = anchor_j + correction_j                 # 0 correction -> Track 1 (UT)

The hidden state `h` is a single GLOBAL vector (PLAN_v1 section 2.1): the per-layer
relabeling symmetry admits only a global or a W-transported carrier, and the global one
is v1's choice. Per-neuron specificity enters only at the readout, through each neuron's
own final-layer features (the alignment etc.); `h_L` supplies shared depth-context.

The readout's last layer is zero-initialized, so at init every track reproduces the UT
champion exactly (do-no-harm, PLAN_v1 section 1). Ablations are applied by the caller,
which multiplies `pooled`/`neuron` by the track's channel mask (dataset.channel_mask)
BEFORE the forward pass -- and retrains (train.py).
"""

import torch
import torch.nn as nn

from dataset import POOLED_DIM, NEURON_DIM

HIDDEN = 128   # power of two so the Monarch factorization is available later


class GlobalRecurrence(nn.Module):
    """Channel 3 update g: sweep depth, carry a single global hidden vector h."""

    def __init__(self, pooled_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        # >>> MONARCH SEAM <<< : this is the only linear map in the recurrence. When the
        # global state goes large (v2 / per-neuron carrier), swap nn.Linear for
        # iterativennsimple.MonarchLinear (same (in_features, out_features) interface).
        self.cell = nn.Linear(pooled_dim + hidden, hidden)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled (B, depth, pooled_dim) -> h_L (B, hidden)
        B, depth, _ = pooled.shape
        h = pooled.new_zeros(B, self.hidden)          # h_0 = 0
        for k in range(depth):
            h = torch.tanh(self.cell(torch.cat([pooled[:, k], h], dim=1)))
        return h


class Corrector(nn.Module):
    """UT-anchored recurrent corrector: estimate = anchor + out_scale * readout."""

    def __init__(self, out_scale: float, pooled_dim: int = POOLED_DIM,
                 neuron_dim: int = NEURON_DIM, hidden: int = HIDDEN):
        super().__init__()
        self.out_scale = out_scale                    # emit corrections in target-std units
        self.rnn = GlobalRecurrence(pooled_dim, hidden)
        self.readout = nn.Sequential(
            nn.Linear(neuron_dim + hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)       # do-no-harm: correction = 0 at init
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, pooled: torch.Tensor, neuron: torch.Tensor,
                anchor: torch.Tensor) -> torch.Tensor:
        # pooled (B, depth, pooled_dim); neuron (B, width, neuron_dim); anchor (B, width)
        h = self.rnn(pooled)                                   # (B, hidden) depth-context
        B, width, _ = neuron.shape
        h_broadcast = h[:, None, :].expand(B, width, h.shape[1])   # (B, width, hidden)
        feat = torch.cat([neuron, h_broadcast], dim=2)             # (B, width, neuron_dim+hidden)
        correction = self.out_scale * self.readout(feat).squeeze(-1)  # (B, width)
        return anchor + correction


if __name__ == "__main__":
    # Smoke test: at init the correction is exactly 0, so estimate == anchor (do-no-harm).
    torch.manual_seed(0)
    B, depth, width = 4, 32, 256
    pooled = torch.randn(B, depth, POOLED_DIM)
    neuron = torch.randn(B, width, NEURON_DIM)
    anchor = torch.rand(B, width)
    model = Corrector(out_scale=2e-3)
    with torch.no_grad():
        est = model(pooled, neuron, anchor)
    print("estimate shape:", tuple(est.shape))
    print("max |estimate - anchor| at init:", float((est - anchor).abs().max()),
          "(should be 0.0 -- zero-init readout)")
    print("params:", sum(p.numel() for p in model.parameters()))
