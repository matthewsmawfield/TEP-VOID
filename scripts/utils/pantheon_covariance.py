#!/usr/bin/env python3
"""
Utility for loading and using the Pantheon+ covariance matrices.

The Pantheon+ data release provides two covariance matrices:
  - Pantheon+SH0ES_STATONLY.cov: statistical only (N×N, N=1701)
  - Pantheon+SH0ES_STAT+SYS.cov: statistical + systematic (N×N, N=1701)

The .cov file format is:
  Line 1: N (number of rows/columns)
  Lines 2..N*N+1: matrix elements, read sequentially (row-major)

The matrix rows/columns correspond to the SNe in Pantheon+SH0ES.dat
in the order they appear in that file (1701 light curves).
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_covariance(cov_path):
    """
    Load a Pantheon+ covariance matrix from .cov file.

    Returns:
        cov: N×N numpy array
        n: matrix dimension
    """
    cov_path = Path(cov_path)
    if not cov_path.exists():
        raise FileNotFoundError(f"Covariance file not found: {cov_path}")

    with open(cov_path) as f:
        n = int(f.readline().strip())
        # Read remaining lines as flat array
        data = np.fromstring(f.read(), sep="\n")

    # Reshape to N×N
    expected = n * n
    if len(data) < expected:
        # Some files may have formatting issues; try parsing differently
        data = np.loadtxt(cov_path, skiprows=1)

    cov = data[:expected].reshape(n, n)
    return cov, n


def get_bin_covariance(cov_full, sn_indices_per_bin, n_total):
    """
    Compute the covariance matrix for binned H0(z) measurements.

    Given the full SN-level covariance matrix and the mapping of SNe
    to bins, compute the bin-level covariance using optimal weighting.

    For each bin, the H0 estimate is a weighted mean of individual SN H0
    values. The bin-level covariance is:

        C_bin[i,j] = sum_{a in bin_i} sum_{b in bin_j} w_a * w_b * C_full[a,b]

    where w_a are the per-SN weights (inverse variance weighting).

    Args:
        cov_full: N×N full covariance matrix
        sn_indices_per_bin: list of arrays, each containing SN indices for one bin
        n_total: total number of SNe (N)

    Returns:
        cov_bin: n_bins × n_bins bin-level covariance matrix
        n_bins: number of bins
    """
    n_bins = len(sn_indices_per_bin)
    cov_bin = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        idx_i = sn_indices_per_bin[i]
        if len(idx_i) == 0:
            continue
        for j in range(i, n_bins):
            idx_j = sn_indices_per_bin[j]
            if len(idx_j) == 0:
                continue

            # Extract sub-matrix
            sub = cov_full[np.ix_(idx_i, idx_j)]

            # Equal weights for now (can be improved with inverse-variance)
            w_i = np.ones(len(idx_i)) / len(idx_i)
            w_j = np.ones(len(idx_j)) / len(idx_j)

            cov_bin[i, j] = w_i @ sub @ w_j
            if i != j:
                cov_bin[j, i] = cov_bin[i, j]

    return cov_bin, n_bins


def compute_chi2_with_covariance(h0_obs, h0_model, cov_bin):
    """
    Compute chi-squared using the full bin-level covariance matrix.

    chi^2 = (h0_obs - h0_model)^T @ C^{-1} @ (h0_obs - h0_model)

    Args:
        h0_obs: observed H0 values per bin
        h0_model: model H0 values per bin
        cov_bin: bin-level covariance matrix

    Returns:
        chi2: chi-squared value
    """
    delta = h0_obs - h0_model
    # Use pseudo-inverse for numerical stability
    try:
        cov_inv = np.linalg.inv(cov_bin)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_bin)

    chi2 = float(delta @ cov_inv @ delta)
    return chi2
