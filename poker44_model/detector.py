"""Poker44 bot detector — C2 + linear-head BLEND (poker-c2-linblend).

Model: a proba-average BLEND of
  (a) the C2 ExtraTrees + HistGradientBoosting soft-vote ensemble (keeps C2's
      benchmark discrimination), and
  (b) an L1-LogisticRegression on StandardScaler(C2 feats) — a LOW-VARIANCE head
      that transfers to the validator-sanitized live population far better than
      the deep trees (which overfit benchmark idiosyncrasy).
score = 0.6 * C2_proba + 0.4 * linear_proba. See blend_model.Blend / train_blend.py.

Everything else is C2 verbatim: the 180 sanitization-invariant features
(features.py), training on hands passed through prepare_hand_for_miner
(train==serve), and WITHIN-BATCH RANK output (matches the ranking-based reward).

IMPORTANT — inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only
TRAINING sanitizes raw benchmark hands. Sanitizing again here would
double-transform, so this path featurizes the incoming chunks directly.

The trained blend is the committed `model.joblib` (a poker44_model.blend_model.Blend).
`score_batch(chunks)` returns one rank-based bot-risk score in [0,1] per chunk.
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES
# Registering the module so joblib can resolve poker44_model.blend_model.Blend on load.
from poker44_model import blend_model  # noqa: F401

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
