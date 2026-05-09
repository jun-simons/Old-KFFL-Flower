"""Toy synthetic dataset for fast KFFL testing.

Generates a binary classification problem where:
  - The true decision boundary depends on a small set of informative features.
  - One binary sensitive attribute (``group``) is partially correlated with the
    label, so a naive model trained to minimise cross-entropy will pick up on
    the correlation, giving the fairness regulariser something to reduce.

Design
------
n=2000 samples, d=10 features, 1 sensitive attribute.

Label generation:
  y = 1  iff  x[0] + x[1] > 0   (linear boundary on first two features)

Sensitive attribute:
  group ~ Bernoulli(0.5)          (balanced groups)
  Correlation with y: P(group=1 | y=1) = 0.7, P(group=1 | y=0) = 0.3
  This means 40% of the variance in ``group`` is explained by y, creating a
  meaningful (but not perfect) fairness signal.

Features:
  x[0], x[1]  ~ N(0, 1)                 — informative, determine label
  x[2]        ~ N(0.4 * group, 0.8)     — partially correlated with group
  x[3..9]     ~ N(0, 1)                 — pure noise
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .adult import DatasetBundle


def load(
    sensitive_features: List[str] | None = None,
    target_feature: str = "label",
    n_samples: int = 2000,
    seed: int = 0,
) -> DatasetBundle:
    """Generate and return the toy synthetic DatasetBundle.

    Parameters
    ----------
    sensitive_features:
        Ignored — the toy dataset always has a single sensitive attribute
        named ``"group"``.  Accepted for API compatibility.
    target_feature:
        Ignored — target is always ``"label"``.
    n_samples:
        Total number of samples.
    seed:
        NumPy random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # ---- Informative features and labels ----------------------------------
    x01 = rng.standard_normal((n_samples, 2))
    y = ((x01[:, 0] + x01[:, 1]) > 0).astype(np.int64)

    # ---- Sensitive attribute (partially correlated with y) ----------------
    # P(group=1 | y=1) = 0.6,  P(group=1 | y=0) = 0.4
    p_group = np.where(y == 1, 0.6, 0.4)
    group = rng.binomial(1, p_group).astype(np.int64)   # shape (n,)

    # ---- Feature matrix ---------------------------------------------------
    # x[0], x[1]: determine y
    # x[2]:       noisy proxy for group (correlated with sensitive attr)
    # x[3..9]:    pure noise
    x2 = 0.4 * group + 0.8 * rng.standard_normal(n_samples)
    x_noise = rng.standard_normal((n_samples, 7))
    X = np.concatenate([x01, x2[:, None], x_noise], axis=1).astype(np.float32)

    # Sensitive matrix: shape (n, 1)
    s = group[:, None].astype(np.int64)

    sensitive_labels: List[Dict[int, str]] = [{0: "group_A", 1: "group_B"}]

    return DatasetBundle(
        X=X,
        y=y,
        s=s,
        feature_names=["x0", "x1", "x2_proxy", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],
        sensitive_names=["group"],
        target_name="label",
        sensitive_labels=sensitive_labels,
    )
