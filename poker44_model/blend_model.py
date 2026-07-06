"""Blend estimator baked into the deployed model.joblib (poker-c2-linblend).

score = w * C2_softvote_proba + (1-w) * linear_proba
  C2_softvote = VotingClassifier([ExtraTrees(300,msl4,n_jobs=1),
                                  HistGB(depth3,lr0.03,300it,l2=1)], soft)
  linear      = LogisticRegression(L1, C) on StandardScaler(C2 feats)

The linear head is a LOW-VARIANCE regularizer that transfers to the sanitized
live population far better than the deep trees (which overfit benchmark
idiosyncrasy); the trees keep C2's benchmark discrimination. Blending gets both.
NO capture-fitting: fit only on sanitized benchmark. All learners n_jobs=1.

This class is imported by joblib at model-load time, so it must live in the
deployed package (poker44_model.blend_model.Blend).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin


def _c2(seed=0):
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4, random_state=seed, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                         l2_regularization=1.0, random_state=seed)
    return VotingClassifier([("et", et), ("hgb", hgb)], voting="soft", n_jobs=1)


def _linear(C=0.25):
    lr = LogisticRegression(max_iter=5000, C=C, penalty="l1", solver="liblinear")
    return make_pipeline(StandardScaler(), lr)


class Blend(BaseEstimator, ClassifierMixin):
    """proba-average blend of C2 trees and an L1-logistic head."""

    def __init__(self, w=0.6, C=0.25, seed=0):
        self.w = w
        self.C = C
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.c2_ = _c2(self.seed).fit(X, y)
        self.lin_ = _linear(self.C).fit(X, y)
        return self

    def predict_proba(self, X):
        c2 = self.c2_.predict_proba(X)[:, 1]
        lin = self.lin_.predict_proba(X)[:, 1]
        s = self.w * c2 + (1.0 - self.w) * lin
        return np.column_stack([1.0 - s, s])
