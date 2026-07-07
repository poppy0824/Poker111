"""Poker44 bot detector — pure-tree candidate `poker-lgb-tuned`.

Model: a **tuned LightGBM gradient-boosted tree** (LGBMClassifier, ~800 trees,
depth 6-8 / num_leaves 63, lr 0.02, subsample/colsample 0.8, L1+L2 reg) over the
SAME 180 sanitization-invariant C2 features (cross-hand duplication `sig_*`,
entropies, structural / aggression aggregates — see features.py FEATURE_NAMES).

This is a PURE GBDT — **no linear head**. New-eval LIVE rounds (R1+R2, 2026-07-07)
showed the L1-logistic head in the linblend miners caps out ~0.40 while pure-tree
scoring reaches 0.55; the linear head hurts ~0.15 under the new eval. lgb_tuned is
the pack's common strong learner in that pure-tree regime, as a distinct-from-C2
(ExtraTrees+HGB) diversification of the tree family.

IMPORTANT — inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand); only
TRAINING sanitizes raw benchmark hands (see train_model.py). Featurize incoming
chunks directly. Output = **within-batch rank** in [0,1] (higher = more bot-like),
matching the validator's ranking-based reward.

LightGBM predict is single-threaded here (num_threads=1 baked into the booster at
train time) so batched predict cannot deadlock the axon. The trained model is the
committed `model.joblib`; joblib/lightgbm load it at inference.
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _raw_scores(model, chunks):
    # Live chunks are already sanitized by the validator; featurize as-is.
    rows = []
    for c in chunks:
        feats = chunk_features(c)          # compute the feature set ONCE per chunk
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model.predict_proba(np.array(rows, dtype=float))[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk model probability (fallback; batch path is score_batch)."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores(_model(), [chunk])[0]), 6)
    except Exception:
        return 0.5
