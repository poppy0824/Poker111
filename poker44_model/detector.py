"""Poker44 bot detector (BEATER rebuild) -- a WITHIN-BATCH RANK-FUSED ENSEMBLE of
three decorrelated members over the 610-dim top-miner UNION feature view
(features_v2 order-statistics + base_features chunk_features), topped with our
reward-fit, FPR-capped floating decision layer.

Why this model
--------------
A fresh full-field head-to-head (every inspectable top miner's model run on the
SAME 1460-group sanitized benchmark under GroupKFold-by-date) found exactly ONE
lever that beats our prior 180-C2 pipeline on true discrimination: the top-miner
UNION feature surface. A plain LGBM over UNION scores GroupKFold-by-date AP 0.943
(per-date 0.932) vs our 180-C2 0.920 (per-date 0.907) -- a real, seed-stable
+0.023. UNION subsumes C2 (C2+UNION = 0.944). Crucially it also un-collapses on
the live captures (raw-STD 0.043, within-batch distinct-fraction 1.00), so it
rides our within-batch rank / calibrated output the same way C2 does. Every other
"edge" in the field (n-gram / temporal-order features, raw-magnitude columns) is
either an order-construction artifact that collapses under hand-shuffle or is
massively OOD -- excluded here on purpose.

Members (all over the identical 610-dim UNION row)
--------------------------------------------------
  1. STACK  -- LGBM + XGB + RF -> logistic OOF stack (the discrimination anchor).
  2. MONO   -- monotone-constrained LightGBM bag on the sign-stable UNION subspace
               (per-DATE Spearman sign stable across >=70% of dates, |rho|>=0.05);
               the OOD-transfer regularizer, decorrelated from the anchor.
  3. MLP    -- StandardScaler -> PCA(56) -> MLP bag; architecturally decorrelated.

Fusion is calibration-free: each member's WITHIN-BATCH rank (argsort/argsort/(n-1))
averaged 0.35/0.30/0.35, so no member's OOD score-scale distorts the blend.

Decision layer (STRICTLY MONOTONE; isotonic removed 2026-07-15)
---------------------------------------------------------------
The fused within-batch rank goes straight into the reward-fit per-batch decision
layer (anchor quantile Q + logit margin/temp + FLOOR + CAP), which SHIFTS each side
of the 0.5 line instead of clamping it -> the map fused -> served is strictly
monotone, so the served order IS the fused order and AP / recall@FPR (the 65% block)
are set purely by the fused rank.

Two corrections vs the previous version, both measured on live captures:
  * The isotonic map is GONE. It is monotone but NON-INJECTIVE, so it merged the
    fused rank into ~22 distinct levels per 100-chunk window and put the
    recall@FPR<=0.05 boundary inside a tie group. The old claim that the transform
    was "monotone, so AP/recall are set purely by the fused rank" was FALSE.
  * FLOOR is 0.10, not 0.02. The old claim of "zero hard-zeros" was ALSO FALSE:
    FLOOR guarantees that k chunks CROSS 0.5, not that any of them is a BOT.
    scoring.py zeroes the WHOLE round when no true bot crosses, and with k=2 the
    crossing set was decided by array index inside the isotonic tie plateau
    (index-arbitrary in 17-18 of 18 live windows) -- which produced uid212 R3 =
    0.000, uid236 R2 = 0.000, and uid236's ~0.077 epoch.
k = ceil(FLOOR*n) chunks cross 0.5; at n=100 that is 10, matching the 10% FPR
budget where threshold_sanity_quality is still 1.0.

IMPORTANT -- inference does NOT sanitize (live chunks arrive already sanitized by
the validator). Only offline training sanitized the raw benchmark hands. All
estimators are pinned single-thread on load (batched-predict deadlock guard).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

try:
    from .union_features import union_features, UNION_NAMES
except ImportError:  # flat-module import at train/eval time
    from union_features import union_features, UNION_NAMES

try:  # keep any torch backend single-threaded
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None


def _pin_single_thread(est):
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception:
            pass
    for attr in ("final_estimator_", "final_estimator"):
        try:
            _pin_single_thread(getattr(est, attr))
        except Exception:
            pass
    try:
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception:
        pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for key in ("stack", "mono", "mlp"):
            try:
                _pin_single_thread(b[key])
            except Exception:
                pass
        _MODEL = b
    return _MODEL


def _rank01(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _rows(chunks):
    rows = []
    for c in chunks:
        feats = union_features(c)
        rows.append([feats.get(k, 0.0) for k in UNION_NAMES])
    return np.array(rows, dtype=float)


def _fused_rank(model, chunks):
    X = _rows(chunks)
    s1 = model["stack"].predict_proba(X)[:, 1]
    s2 = model["mono"].predict_proba(X)[:, 1]
    s3 = model["mlp"].predict_proba(X)[:, 1]
    w1, w2, w3 = model["weights"]
    return (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


_T_HI = 0.00040000000000000002   # logit(0.5001): sigmoid(t) >= 0.5001 <=> t >= _T_HI
_T_LO = -0.00040000000000000002  # logit(0.4999): sigmoid(t) <= 0.4999 <=> t <= _T_LO


def _decision(model, v):
    """Reward-fit, FPR-capped per-batch decision layer on the TIE-FREE fused rank.

    Identical to the deployed layer (same Q / MARGIN / TEMP / FLOOR / CAP / EPS /
    train_ref_logit, same k, same crossing count) except for two tie sources that
    were destroying the 65% rank block (0.35*AP + 0.30*recall@FPR<=0.05, both of
    which argsort the served scores and break ties by ARRAY INDEX):

      1. the isotonic map is GONE -- it is monotone but NON-INJECTIVE, so it
         merged the fused rank into ~26 distinct levels per 100-chunk window and
         put the recall@FPR<=0.05 boundary INSIDE a tie group;
      2. FLOOR/CAP now SHIFT each side instead of CLAMPing it to the constants
         0.5001 / 0.4999, which preserves the internal spacing of both groups.

    The result is a STRICTLY MONOTONE map fused -> served score, so the served
    order is exactly the model's order, while k = ceil(FLOOR*n) chunks still
    cross 0.5 (FLOOR lifts the top-k, CAP pins the rest below) -- the 30%
    hard-0.5-threshold block is unchanged.
    """
    eps = float(model["EPS"])
    q = float(model["Q"])
    margin = float(model["MARGIN"])
    temp = float(model.get("TEMP", 1.0))
    floor = float(model["FLOOR"])
    cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(v, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    t = (z - anchor + tref) / temp
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(t))))
    top, rest = order[:k], order[k:]
    # FLOOR (tie-free): shift the top-k as a block so its MINIMUM sits at 0.5001
    # -- never an all-below-0.5 hard zero, but the spacing inside the block (and
    # hence the ordering that AP / bot-recall read) survives.
    d = _T_HI - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if cap and rest.size:
        # CAP (tie-free): shift the rest as a block so its MAXIMUM sits at 0.4999
        # -> deterministic crossing count k, spacing preserved.
        d = t[rest].max() - _T_LO
        if d > 0.0:
            t[rest] = t[rest] - d
    scores = 1.0 / (1.0 + np.exp(-t))
    return [round(float(s), 9) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (rank-fused, reward-fit floating output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _fused_rank(m, chunks))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        X = _rows([chunk])
        s = (m["weights"][0] * m["stack"].predict_proba(X)[:, 1]
             + m["weights"][1] * m["mono"].predict_proba(X)[:, 1]
             + m["weights"][2] * m["mlp"].predict_proba(X)[:, 1]) / sum(m["weights"])
        return round(float(s[0]), 6)
    except Exception:
        return 0.5
