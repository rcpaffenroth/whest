"""dataset.py -- assemble the corrector's training tensors from the HF phase-1 MLPs.

For each of the 100 public MLPs we run the frozen UT (features.py) and build, per MLP:
  anchor : (width,)            UT final-layer mean -- the base; estimate = anchor + corr
  target : (width,)            truth - anchor -- what the correction must equal
  pooled : (depth, POOLED_DIM) per-layer pooled features for the depth recurrence
  neuron : (width, NEURON_DIM) per-neuron final-layer features for the readout

Features carry three PROVENANCE groups, masked per track (PLAN_v1 sections 1, 3):
  cloud-only   post-activation moments        needs the cloud     -> mask by c_on
  W-only       participation ratio            needs W             -> mask by w_on
  W x cloud    pre moments, actprob, rect      needs BOTH          -> mask by w_on*c_on
So Track 2 = (c_on=1, w_on=0), Track 3 = (0,1), Track 4 = (1,1); the interaction
features exist only when both ports are on. Ablation is by RE-TRAINING with the mask
applied (train.py), not eval-time masking.

Normalization is fit on the TRAIN split only. Masking a (standardized) group to 0 sets
it to its mean = "no information". No disk cache -- one extraction (~30 s) per process;
train.py calls build_dataset once.
"""

import os

import numpy as np
import torch
from datasets import load_dataset

from features import extract_features

# The numpy UT sweep over 100 MLPs takes several minutes (float64 moments dominate), so
# cache the raw per-MLP features and recompute only when n_rot changes. Split/normalize
# are cheap and always redone from the cache.
CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")


# --- feature layout ---------------------------------------------------------------
# Pooled per-layer vector (POOLED_DIM = 22), by provenance group:
#   [0:8]   cloud-only : pool over neurons of post moments (mean,std,skew,kurt) -> 2*4
#   [8:10]  W-only     : pool of participation ratio                            -> 2*1
#   [10:22] W x cloud  : pool of [pre(4), actprob, rect]                         -> 2*6
POOLED_DIM = 22
POOLED_CLOUD = slice(0, 8)
POOLED_WONLY = slice(8, 10)
POOLED_INTER = slice(10, 22)

# Per-neuron final vector (NEURON_DIM = 11), by provenance group:
#   [0:1]   always     : anchor (UT final mean) -- available to every track
#   [1:4]   cloud-only : post std, skew, kurt   (mean is the anchor, above)
#   [4:5]   W-only     : participation ratio
#   [5:11]  W x cloud  : pre(4)=alignment,std,skew,kurt ; actprob ; rect
NEURON_DIM = 11
NEURON_CLOUD = slice(1, 4)
NEURON_WONLY = slice(4, 5)
NEURON_INTER = slice(5, 11)


def _pool_over_neurons(x):
    """Permutation-invariant pool over the neuron axis (axis 1); respects the per-layer
    relabeling symmetry (PLAN_v1 section 2.1). x (depth,width[,k]) -> (depth, 2k)."""
    d = x.shape[0]
    return np.concatenate([x.mean(1).reshape(d, -1), x.std(1).reshape(d, -1)], axis=1)


def _assemble_one(rec):
    """One MLP's feature record -> (pooled (depth,22), neuron (width,11)) raw features."""
    post, pre = rec["post"], rec["pre"]            # (depth, width, 4)
    actprob, rect, pr = rec["actprob"], rec["rect"], rec["pr"]  # (depth, width)
    inter = np.concatenate([pre, actprob[..., None], rect[..., None]], axis=2)  # (depth,width,6)

    pooled = np.concatenate([
        _pool_over_neurons(post),          # cloud-only  (depth, 8)
        _pool_over_neurons(pr),            # W-only      (depth, 2)
        _pool_over_neurons(inter),         # W x cloud   (depth, 12)
    ], axis=1)                             # (depth, 22)

    anchor = rec["ut_final_mean"]          # (width,)
    neuron = np.concatenate([
        anchor[:, None],                   # always      (width, 1)
        post[-1, :, 1:4],                  # cloud-only  (width, 3): std, skew, kurt
        pr[-1][:, None],                   # W-only      (width, 1)
        inter[-1],                         # W x cloud   (width, 6)
    ], axis=1)                             # (width, 11)
    return pooled, neuron


def channel_mask(w_on: int, c_on: int):
    """0/1 masks (pooled (22,), neuron (11,)) for a track. Interaction group needs BOTH."""
    pm = np.ones(POOLED_DIM, dtype=np.float32)
    pm[POOLED_CLOUD] = c_on
    pm[POOLED_WONLY] = w_on
    pm[POOLED_INTER] = w_on * c_on
    nm = np.ones(NEURON_DIM, dtype=np.float32)   # NEURON[0] (anchor) always on
    nm[NEURON_CLOUD] = c_on
    nm[NEURON_WONLY] = w_on
    nm[NEURON_INTER] = w_on * c_on
    return torch.from_numpy(pm), torch.from_numpy(nm)


def build_dataset(n_rot: int = 13, n_train: int = 80, seed: int = 0):
    """Extract features for all 100 phase-1 MLPs, split, normalize on train.

    Returns a dict of torch float32 tensors and metadata:
      pooled/neuron/anchor/target, each with _train (n_train,...) and _val (100-n_train,...)
      out_scale : scalar -- correction is emitted in units of the train target std
    """
    cache = os.path.join(CACHE_DIR, f"feats_nrot{n_rot}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        pooled, neuron, anchor, target = z["pooled"], z["neuron"], z["anchor"], z["target"]
    else:
        ds = load_dataset(
            "aicrowd/arc-whestbench-public-2026", "default", revision="v1-phase1", split="mini"
        )
        pooled, neuron, anchor, target = [], [], [], []
        for i, ex in enumerate(ds):
            W = np.asarray(ex["weights"], dtype=np.float32)
            truth = np.asarray(ex["final_means"], dtype=np.float64)
            rec = extract_features(W, ex["mlp_seed"], n_rot=n_rot)
            p, n = _assemble_one(rec)
            pooled.append(p)
            neuron.append(n)
            anchor.append(rec["ut_final_mean"])
            target.append(truth - rec["ut_final_mean"])   # what the correction must equal
            print(f"\rextracting features: {i + 1}/{len(ds)} MLPs", end="", flush=True)
        print()
        pooled = np.stack(pooled).astype(np.float32)   # (100, depth, 22)
        neuron = np.stack(neuron).astype(np.float32)   # (100, width, 11)
        anchor = np.stack(anchor).astype(np.float32)   # (100, width)
        target = np.stack(target).astype(np.float32)   # (100, width)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, pooled=pooled, neuron=neuron, anchor=anchor, target=target)

    # deterministic split
    idx = np.random.default_rng(seed).permutation(len(pooled))
    tr, va = idx[:n_train], idx[n_train:]

    # standardize pooled/neuron per feature-column using TRAIN stats only
    def fit(x):  # x (N, ..., D) -> per-D mean/std over all but last axis of TRAIN
        flat = x[tr].reshape(-1, x.shape[-1])
        mu = flat.mean(0)
        sd = flat.std(0)
        sd = np.where(sd > 0, sd, 1.0)   # a constant feature -> std 0; leave it centered
        return mu, sd

    pmu, psd = fit(pooled)
    nmu, nsd = fit(neuron)
    pooled = (pooled - pmu) / psd
    neuron = (neuron - nmu) / nsd

    out_scale = float(target[tr].std())   # emit corrections in train-target-std units

    T = torch.from_numpy
    return dict(
        pooled_train=T(pooled[tr]), pooled_val=T(pooled[va]),
        neuron_train=T(neuron[tr]), neuron_val=T(neuron[va]),
        anchor_train=T(anchor[tr]), anchor_val=T(anchor[va]),
        target_train=T(target[tr]), target_val=T(target[va]),
        out_scale=out_scale, n_rot=n_rot,
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")  # quiet the HF token notice
    d = build_dataset()
    for k in ("pooled_train", "neuron_train", "anchor_train", "target_train",
              "pooled_val", "neuron_val"):
        print(f"{k:16s} {tuple(d[k].shape)}")
    print(f"out_scale (train target std): {d['out_scale']:.3e}")
    # base (zero-correction) MSE should equal the UT MSE on each split
    print(f"train UT MSE (=target var):   {float((d['target_train']**2).mean()):.3e}")
    print(f"val   UT MSE (=target var):   {float((d['target_val']**2).mean()):.3e}")
    for name, (w, c) in {"Track2": (0, 1), "Track3": (1, 0), "Track4": (1, 1)}.items():
        pm, nm = channel_mask(w, c)
        print(f"{name}: pooled active {int(pm.sum())}/{POOLED_DIM}, "
              f"neuron active {int(nm.sum())}/{NEURON_DIM}")
