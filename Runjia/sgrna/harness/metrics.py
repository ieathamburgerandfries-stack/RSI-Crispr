"""Small numeric helpers that several experiments need and that used to be copied into each of them.

    zscore(p)   (p - mean) / (sd + 1e-12): the scale on which predictions of different models are averaged into an ensemble
"""
from __future__ import annotations


def zscore(p):
    """Standardise a vector of predictions (the 1e-12 keeps a constant vector from dividing by zero)."""
    return (p - p.mean()) / (p.std() + 1e-12)
