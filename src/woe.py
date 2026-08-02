"""
Weight of Evidence (WOE) and Information Value (IV) utilities for credit scoring.

Usage:
    table, iv, bin_edges = calc_woe_iv(train_df, 'int_rate', 'default', bins=10)
    woe_series = apply_woe(test_df, 'int_rate', table, is_categorical=False, bin_edges=bin_edges)
"""

import numpy as np
import pandas as pd


def calc_woe_iv(df, feature, target, bins=10, is_categorical=False):
    """
    Compute WOE and IV for a single feature, fit on the given dataframe.

    IMPORTANT: fit only on the training set. Reuse the returned bin_edges
    (for continuous features) to transform any other dataset (e.g. test).

    Low-cardinality numeric features (e.g. term, pub_rec with few unique
    values) should be passed with is_categorical=True -- pd.qcut on a
    low-cardinality column can silently collapse into a single bin and
    produce a falsely-zero IV.

    Returns:
        grouped (pd.DataFrame): per-bin counts, WOE, and IV contribution
        total_iv (float): summed IV across all bins
        bin_edges (np.ndarray or None): bin edges for continuous features,
            None for categorical features
    """
    df = df[[feature, target]].copy()
    bin_edges = None

    if is_categorical:
        df['bin'] = df[feature].astype(str)
    else:
        df['bin'], bin_edges = pd.qcut(df[feature], bins, duplicates='drop', retbins=True)

    grouped = df.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    grouped['pct_good'] = (grouped['good'] / grouped['good'].sum()).replace(0, 0.0001)
    grouped['pct_bad'] = (grouped['bad'] / grouped['bad'].sum()).replace(0, 0.0001)

    grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    grouped['iv'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']

    return grouped, grouped['iv'].sum(), bin_edges


def apply_woe(df, feature, woe_table, is_categorical, bin_edges=None):
    """
    Map raw feature values to their WOE score, using bins/edges learned
    from the training set. Unseen categories or out-of-range values map
    to neutral WOE (0).
    """
    if is_categorical:
        woe_map = woe_table['woe'].to_dict()
        result = df[feature].astype(str).map(woe_map)
    else:
        binned = pd.cut(df[feature], bins=bin_edges, include_lowest=True)
        woe_map = woe_table['woe'].to_dict()
        result = binned.map(woe_map)

    return result.astype(float).fillna(0)


def build_woe_dataset(df, features, categorical_features, woe_tables, bin_edges_map):
    """
    Build a full WOE-transformed feature matrix for a set of features.
    """
    woe_df = pd.DataFrame(index=df.index)
    for feat in features:
        is_cat = feat in categorical_features
        edges = bin_edges_map.get(feat) if not is_cat else None
        woe_df[feat + '_woe'] = apply_woe(df, feat, woe_tables[feat], is_cat, edges)
    return woe_df
