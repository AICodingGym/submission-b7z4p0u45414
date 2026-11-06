"""Shared data loading and fold assignment (used by train_lstm_v2.py and ensemble.py).
Fold assignment is byte-for-byte the same logic as train_lstm.py v1 (lines 83-87), so v1 and v2 OOF files line up.
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.expanduser('~/ventilator-pressure-prediction')


def load_train():
    """Train sorted by (breath_id, time_step) so reshape(-1, 80) gives one breath per row."""
    return pd.read_csv(os.path.join(ROOT, 'data/train.csv')).sort_values(['breath_id', 'time_step'])


def make_folds(tr, nfolds=5):
    """One fold label per breath, stratified by (R, C). Returns (bid, fold_of) in reshape(-1, 80) row order."""
    bid = tr.breath_id.values.reshape(-1, 80)[:, 0]
    rc = tr.groupby('breath_id')[['R', 'C']].first().loc[bid]
    fold_of = np.zeros(len(bid), int)
    rng = np.random.RandomState(0)
    for _, idx in pd.Series(np.arange(len(bid))).groupby((rc.R * 100 + rc.C).values):
        idx = idx.values.copy(); rng.shuffle(idx); fold_of[idx] = np.arange(len(idx)) % nfolds
    return bid, fold_of


def targets(tr):
    """(Y, M): pressure and scored-step mask (u_out == 0), shape (n_breaths, 80)."""
    return (tr.pressure.values.astype('float32').reshape(-1, 80),
            (tr.u_out.values == 0).reshape(-1, 80))
