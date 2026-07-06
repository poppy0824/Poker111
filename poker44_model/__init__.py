"""Participant-owned model package for the Poker44 miner — poker-c2-linblend.

Bot detector = a proba-average BLEND of the C2 ExtraTrees+HistGradientBoosting
soft-vote ensemble (benchmark discrimination) and an L1-logistic head on the same
StandardScaler(C2 feats) (low-variance transfer to the sanitized live feed).
score = 0.6*C2 + 0.4*linear. Same 180 sanitization-invariant features as C2,
trained on hands passed through prepare_hand_for_miner (train==serve), scored by
within-batch ranking. Inference does NOT re-sanitize (live hands are already
sanitized validator-side). See detector.py (inference), blend_model.py (the Blend
estimator), features.py (extraction + FEATURE_NAMES), train_blend.py (training),
model.joblib (trained blend).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
