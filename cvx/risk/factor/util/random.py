# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd


def random_weights(assets):
    """
    Construct a vector of non-negative random weights. Their sum shall be 1
    """
    # Get some random weights
    weights = pd.Series(index=assets, data=np.random.rand(len(assets)))
    return weights / weights.sum()


def random_factors(T, N=2, const_factor=True):
    """
    Construct N random factor time series for T timestamps
    """
    factors = pd.DataFrame(
        index=range(1, T + 1),
        columns=[f"F{i}" for i in range(N)],
        data=np.random.randn(T, N),
    )
    # add the constant factor
    if const_factor:
        factors["const"] = 1
    return factors


def random_beta(assets, factors):
    """
    Construct a random exposure matrix
    """
    data = np.random.randn(factors.shape[1], len(assets))
    return pd.DataFrame(columns=assets, index=factors.columns, data=data)


def random_noise(frame):
    """
    Construct a frame of random noise with exactly the same dimensions as the input frame
    """
    return pd.DataFrame(
        columns=frame.columns,
        index=frame.index,
        data=np.random.randn(frame.shape[0], frame.shape[1]),
    )


def random_assets(num):
    """
    Construct a vector of random assets
    """
    return [str(uuid.uuid4())[:7] for _ in range(num)]
