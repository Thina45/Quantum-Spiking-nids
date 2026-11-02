#!/usr/bin/env python3
"""Write a copy of a processed bundle with a stratified-subsampled test set,
so the noise sweep (slow density-matrix simulation per noisy level) runs on
a few thousand points instead of the full test split."""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = dict(np.load(a.bundle, allow_pickle=True))
    y = d["y_test"]
    rng = np.random.RandomState(a.seed)

    classes, counts = np.unique(y, return_counts=True)
    frac = a.n / len(y)
    idx_parts = []
    for c, cnt in zip(classes, counts):
        take = max(1, int(round(cnt * frac)))
        cls_idx = np.where(y == c)[0]
        idx_parts.append(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)

    d["X_test"] = d["X_test"][idx]
    d["y_test"] = d["y_test"][idx]

    np.savez(a.out, **d)
    print(f"[io] subsampled test set {len(y)} -> {len(idx)} -> {a.out}")


if __name__ == "__main__":
    main()
