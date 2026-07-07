"""Participant-owned model package for the Poker44 miner — pure-tree `poker-lgb-tuned`.

Bot detector = a tuned LightGBM GBDT (~800 trees, num_leaves 63, lr 0.02,
subsample/colsample 0.8, L1+L2 reg) over the SAME 180 sanitization-invariant C2
features. Pure gradient-boosted trees, NO linear head (the new-eval live signal:
pure trees reach ~0.55, the linblend linear head caps ~0.40). Trained on the full
sanitized benchmark passed through the validator's prepare_hand_for_miner
(train==serve); scored by within-batch ranking. Inference does NOT re-sanitize
(live hands are already sanitized validator-side). LightGBM predict is
single-threaded (num_threads=1 baked in) so it cannot deadlock the axon. See
detector.py (inference), features.py (extraction + FEATURE_NAMES), train_model.py
(training), model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
