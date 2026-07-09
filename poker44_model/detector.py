"""Poker44 bot detector — MLP-bag over C2's 180 sanitization-invariant features.

Replaces a tree ensemble with a bag of standardized Torch MLPs (mlp_bag.BagMLP /
mlp_member.TorchMLPClassifier) over the same 180 features. Inputs are standardized
on the train mean/std, each member early-stops on validation loss, and the members
are averaged. Output is a within-batch rank-anchored logistic: the bag probability
is mapped to its within-batch rank u in [0,1], then
    score = sigmoid( TEMP * (u - (1 - BOT_FRACTION)) )
so the top BOT_FRACTION of each batch lands above 0.5. Rank-preserving and
level-invariant.

Inference does NOT sanitize: live chunks arrive already sanitized by the validator
(prepare_hand_for_miner runs validator-side, per hand); only training sanitizes.
"""
from __future__ import annotations

import os

import numpy as np

try:  # bound CPU threads so batched predict stays fast and never deadlocks
    import torch
    torch.set_num_threads(int(os.environ.get("POKER44_TORCH_THREADS", "1")))
except Exception:
    pass

import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

BOT_FRACTION = 0.15
TEMP = 22.0

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank01(vals):
    """Within-batch rank in [0,1] (0 = lowest bag prob, 1 = highest)."""
    n = len(vals)
    if n <= 1:
        return np.array([1.0] * n)
    order = np.argsort(np.argsort(np.asarray(vals, dtype=float), kind="mergesort"))
    return order / (n - 1)


def _rank_anchored_logistic(vals):
    """Top BOT_FRACTION of the batch crosses 0.5; monotone in the bag probability."""
    u = _rank01(vals)
    scores = 1.0 / (1.0 + np.exp(-TEMP * (u - (1.0 - BOT_FRACTION))))
    if scores.size and float(np.max(scores)) < 0.5:
        scores[int(np.argmax(u))] = 0.5
    return [round(float(s), 6) for s in scores]


def _raw_scores(model, chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model.predict_proba(np.array(rows, dtype=float))[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (rank-anchored logistic output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_anchored_logistic(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; the batch path (score_batch) is the real entry."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores(_model(), [chunk])[0]), 6)
    except Exception:
        return 0.5
