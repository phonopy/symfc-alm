"""Tests of RidgeRegression class."""

import numpy as np
import pytest

from symfc_alm import (
    CellDataset,
    DispForceDataset,
    RidgeRegression,
    SymfcAlm,
    standardize_data,
)


@pytest.fixture
def regression():
    """Get RidgeRegression class."""
    return RidgeRegression()


@pytest.fixture
def data(si_111_dataset: DispForceDataset, si_111_structure: CellDataset):
    """Using Si fc3, generate matrix A from displacements and vector b from forces."""
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=2)
    return A, b


def test_run(regression: RidgeRegression, data):
    """Test RidgeRegression.run()."""
    A, b = data
    alpha = 0.1
    regression.run(A, b, alpha)
    assert regression.coeff is not None
    assert regression.psi is not None
    assert regression.errors is not None


def test_run_auto(regression: RidgeRegression, data):
    """Test RidgeRegression.run_auto()."""
    A, b = data
    regression.run_auto(A, b, min_alpha=-2, max_alpha=0, n_alphas=3)
    assert regression.alphas is not None
    assert regression.opt_alpha is not None
    assert regression.coeff is not None
    assert regression.psi is not None
    assert regression.errors is not None


def test_fit(regression: RidgeRegression, data):
    """Test RidgeRegression._fit()."""
    A, b = data
    alpha = 0.1
    regression._fit(A, b, alpha)
    assert regression.coeff is not None


def test_predict(regression: RidgeRegression, data):
    """Test RidgeRegression._predict()."""
    A, b = data
    alpha = 0.1
    regression._fit(A, b, alpha)
    prediction = regression._predict(A)
    np.testing.assert_array_equal(prediction.shape, b.shape)


def test_calc_error(regression: RidgeRegression, data):
    """Test RidgeRegression._calc_error()."""
    A, b = data
    alpha = 0.1
    regression._fit(A, b, alpha)
    error = regression._calc_error(A, b, alpha)
    assert isinstance(error, float)


def test_standardize_data(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset
):
    """Test standardize_data()."""
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, _ = sfa.get_matrix_elements(maxorder=2)
    std_A, scale = standardize_data(A)
    np.testing.assert_array_equal(std_A.shape, A.shape)
    np.testing.assert_array_equal(scale.shape[0], A.shape[1])
