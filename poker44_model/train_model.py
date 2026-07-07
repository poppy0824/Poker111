"""Reproducible training for the pure-tree `poker-lgb-tuned` -> writes model.joblib.

The SAME 180 sanitization-invariant C2 features (features.py FEATURE_NAMES),
scored by a TUNED LightGBM gradient-boosted tree (LGBMClassifier). Pure GBDT —
no linear head (the new-eval live signal favours pure trees over the linblend
linear head). A distinct-from-C2 (ExtraTrees+HGB) diversification within the
tree family and the pack's common strong learner.

The KEY train==serve fix (inherited from C2): every raw benchmark hand is passed
through the validator's `prepare_hand_for_miner` (payload_view.py) BEFORE feature
extraction, so the training distribution matches what the validator serves miners.
Live chunks are already sanitized validator-side, so inference does NOT
re-sanitize — only training does.

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw \
        --payload-view /root/ares/Poker/main/poker44/validator/payload_view.py

n_jobs / num_threads is forced to 1 (ExtraTrees n_jobs=-1 deadlocks the axon;
LightGBM multithread predict is likewise pinned single-thread and baked into the
booster so batched inference stays deterministic and deadlock-free).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import typing

import numpy as np
import joblib
from lightgbm import LGBMClassifier

from poker44_model.features import chunk_features, FEATURE_NAMES

# Tuned pure-GBDT hyper-parameters (single-thread, no linear head).
LGB_PARAMS = dict(
    n_estimators=800,
    max_depth=7,
    num_leaves=63,
    learning_rate=0.02,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=0,
    n_jobs=1,
    num_threads=1,
    verbosity=-1,
)


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional  # payload_view uses Optional but never imports it
    fn = pv.prepare_hand_for_miner

    def sanitize_chunk(chunk):
        out = []
        for h in (chunk or []):
            try:
                out.append(fn(h))
            except Exception:
                out.append(h)
        return out

    return sanitize_chunk


def load(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def build_model(seed=0):
    p = dict(LGB_PARAMS)
    p["random_state"] = seed
    return LGBMClassifier(**p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to train/raw chunk JSON dir")
    ap.add_argument("--payload-view", required=True,
                    help="path to poker44/validator/payload_view.py (the sanitizer)")
    args = ap.parse_args()

    sanitize_chunk = _load_sanitizer(args.payload_view)

    data = load(args.data)
    rows, y = [], []
    for g, l in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE: sanitize raw hands
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
    X = np.array(rows, dtype=float)
    y = np.array(y)

    model = build_model(seed=0).fit(X, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features)")


if __name__ == "__main__":
    main()
