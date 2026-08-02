"""
Convert a WOE-based logistic regression into a points-based credit scorecard.

Standard scaling: `factor` points to double the odds, anchored at `offset`
points for a target odds ratio. Default here follows a common convention
(20 points to double the odds), then results are typically rescaled to a
more familiar range (e.g. mean 600 / std 100) via rescale_score().
"""

import numpy as np
import pandas as pd


def build_scorecard(woe_tables, coefs, intercept, n_features, factor, offset):
    """
    Convert per-feature WOE tables + logistic regression coefficients into
    a points table per feature/bin.

    coefs: dict mapping '<feature>_woe' -> logistic regression coefficient
    """
    scorecard = {}
    for feat_woe_name, coef in coefs.items():
        feat = feat_woe_name.replace('_woe', '')
        table = woe_tables[feat]
        points = -(table['woe'] * coef + intercept / n_features) * factor + offset / n_features
        scorecard[feat] = points
    return scorecard


def score_from_woe(X_woe, coefs, intercept, n_features, factor, offset):
    """
    Vectorized scoring: compute total points per row directly from an
    already WOE-transformed feature matrix (faster than looking up bins
    row by row).
    """
    score = pd.Series(0.0, index=X_woe.index)
    for feat_woe_name, coef in coefs.items():
        score += -(X_woe[feat_woe_name] * coef + intercept / n_features) * factor + offset / n_features
    return score


def rescale_score(score, current_mean, current_std, target_mean, target_std):
    """
    Linearly rescale a score series to a target mean/std (e.g. 600/100).
    This preserves rank-ordering and all discrimination metrics (AUC, KS,
    Gini) exactly -- it only changes presentation.

    Fit current_mean/current_std on the TRAINING set and reuse them when
    rescaling any other dataset (e.g. test), to avoid leaking test
    distribution info into the transformation.
    """
    return target_mean + (score - current_mean) / current_std * target_std
