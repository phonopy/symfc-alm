"""Tests of RidgeRegression class."""

import numpy as np
import pytest

from symfc_alm import CellDataset, DispForceDataset, SymfcAlm
from symfc_alm.ridge import RidgeRegression, ridge_regression, standardize_data


@pytest.fixture
def data(si_111_dataset: DispForceDataset, si_111_structure: CellDataset):
    """Using Si fc3, generate matrix A from displacements and vector b from forces."""
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=2)
    return A, b


@pytest.fixture
def regression(data):
    """Get RidgeRegression class using Si fc3."""
    A, b = data
    return RidgeRegression(A, b, standardize=True)


def test_ridge_regression(data):
    """Test ridge_regression()"""
    alpha = 0.1
    A, b = data
    psi = ridge_regression(A, b, alpha, auto=False)
    assert psi is not None


def test_ridge_regression_with_sklearn(data):
    """Test RidgeRegression.run() compared with scikit-learn."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    alpha = 0.1
    A, b = data
    # use scikit-learn
    ridge_sk = Ridge(alpha=alpha, fit_intercept=False)
    scaler = StandardScaler().fit(A)
    ridge_sk.fit(scaler.transform(A), b)
    fc_ridge_sk = np.true_divide(ridge_sk.coef_, scaler.scale_)
    # use our method
    fc_ridge = ridge_regression(A, b, alpha, auto=False)
    np.testing.assert_allclose(fc_ridge_sk, fc_ridge)


def test_run(regression: RidgeRegression):
    """Test RidgeRegression.run()."""
    alpha = 0.1
    regression.run(alpha)
    assert regression.coeff is not None
    assert regression.psi is not None
    assert regression.errors is not None


def test_run_with_sklearn(regression: RidgeRegression, data):
    """Test RidgeRegression.run() compared with scikit-learn."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    alpha = 0.1
    A, b = data
    # use scikit-learn
    ridge_sk = Ridge(alpha=alpha, fit_intercept=False)
    scaler = StandardScaler().fit(A)
    ridge_sk.fit(scaler.transform(A), b)
    fc_ridge_sk = np.true_divide(ridge_sk.coef_, scaler.scale_)
    # use our method
    regression.run(alpha=alpha)
    fc_ridge = regression.psi
    np.testing.assert_allclose(fc_ridge_sk, fc_ridge)


def test_run_auto(regression: RidgeRegression):
    """Test RidgeRegression.run_auto()."""
    regression.run_auto(min_alpha=-2, max_alpha=0, n_alphas=3)
    assert regression.alphas is not None
    assert regression.opt_alpha is not None
    assert regression.coeff is not None
    assert regression.psi is not None
    assert regression.errors is not None


def test_fit(regression: RidgeRegression):
    """Test RidgeRegression._fit()."""
    alpha = 0.1
    regression._fit(alpha)
    assert regression.coeff is not None


def test_predict(regression: RidgeRegression, data):
    """Test RidgeRegression._predict()."""
    alpha = 0.1
    A, b = data
    regression._fit(alpha)
    prediction = regression._predict(A)
    np.testing.assert_array_equal(prediction.shape, b.shape)


def test_calc_error(regression: RidgeRegression):
    """Test RidgeRegression._calc_error()."""
    alpha = 0.1
    regression._fit(alpha)
    error = regression._calc_error(alpha)
    assert isinstance(error, float)


def test_standardize_data(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset, data
):
    """Test standardize_data()."""
    A, _ = data
    std_A, scale = standardize_data(A)
    np.testing.assert_array_equal(std_A.shape, A.shape)
    np.testing.assert_array_equal(scale.shape[0], A.shape[1])
