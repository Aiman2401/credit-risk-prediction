"""
Credit-risk-specific validation metrics: KS statistic, Population Stability
Index (PSI), and calibration helpers.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calc_ks(probs, y_true):
    """
    Kolmogorov-Smirnov statistic: the maximum separation between the
    cumulative distributions of predicted probability for actual bads vs.
    actual goods. Rule of thumb: <20 weak, 20-30 acceptable, 30-40 good,
    40+ strong (industry convention, not a hard rule).
    """
    ks_stat, p_value = ks_2samp(probs[y_true == 1], probs[y_true == 0])
    return ks_stat, p_value


def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index between two distributions (e.g. train vs.
    test score or feature distribution).

    Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift (monitor),
    >0.25 significant shift (investigate / consider retraining).
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pct = np.percentile(expected, breakpoints)
    expected_pct[0] = -np.inf
    expected_pct[-1] = np.inf

    expected_counts = np.histogram(expected, bins=expected_pct)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=expected_pct)[0] / len(actual)

    expected_counts = np.where(expected_counts == 0, 0.0001, expected_counts)
    actual_counts = np.where(actual_counts == 0, 0.0001, actual_counts)

    psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
    return psi


def gini_from_auc(auc):
    """Gini coefficient, derived directly from AUC."""
    return 2 * auc - 1


def default_rate_by_band(df, score_col, target_col, bands=10):
    """
    Bucket applicants into score bands (quantiles) and compute the actual
    observed default rate per band -- the standard scorecard validation
    chart (should be monotonically decreasing from low score to high).
    """
    bands_series = pd.qcut(df[score_col], bands, duplicates='drop')
    return df.groupby(bands_series, observed=True)[target_col].mean().sort_index()
