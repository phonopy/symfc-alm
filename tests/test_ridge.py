"""Tests of RidgeRegression class."""

import numpy as np
import pytest

from symfc_alm.ridge import RidgeRegression, ridge_regression, standardize_data


def test_ridge_regression(si_111_Ab):
    """Test ridge_regression()."""
    alpha = 0.1
    A, b = si_111_Ab
    psi = ridge_regression(A, b, alpha, auto=False)
    assert psi is not None


@pytest.mark.big
def test_ridge_regression_deficient(aln_332_Ab):
    """Test ridge_regression()."""
    alpha = 0.1
    A, b = aln_332_Ab
    psi = ridge_regression(A, b, alpha, auto=False)
    assert psi is not None


def test_ridge_regression_with_sklearn(si_111_Ab):
    """Test RidgeRegression.run() compared with scikit-learn."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    alpha = 0.1
    A, b = si_111_Ab
    # use scikit-learn
    ridge_sk = Ridge(alpha=alpha, fit_intercept=False)
    scaler = StandardScaler().fit(A)
    ridge_sk.fit(scaler.transform(A), b)
    fc_ridge_sk = np.true_divide(ridge_sk.coef_, scaler.scale_)
    # use our method
    fc_ridge = ridge_regression(A, b, alpha, auto=False)
    np.testing.assert_allclose(fc_ridge_sk, fc_ridge, rtol=1e-04, atol=1e-06)


def test_run(si_111_Ab):
    """Test RidgeRegression.run()."""
    alpha = 0.1
    A, b = si_111_Ab
    regression = RidgeRegression(A, b, standardize=True)
    regression.run(alpha)
    assert regression.coeff is not None
    assert regression.psi is not None
    assert regression.errors is not None


def test_run_with_sklearn(si_111_Ab):
    """Test RidgeRegression.run() compared with scikit-learn."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    alpha = 0.1
    A, b = si_111_Ab
    # use scikit-learn
    ridge_sk = Ridge(alpha=alpha, fit_intercept=False)
    scaler = StandardScaler().fit(A)
    ridge_sk.fit(scaler.transform(A), b)
    fc_ridge_sk = np.true_divide(ridge_sk.coef_, scaler.scale_)
    # use our method
    regression = RidgeRegression(A, b, standardize=True)
    regression.run(alpha=alpha)
    fc_ridge = regression.psi
    np.testing.assert_allclose(fc_ridge_sk, fc_ridge, rtol=1e-04, atol=1e-06)


def test_run_auto(si_111_Ab):
    """Test RidgeRegression.run_auto()."""
    A, b = si_111_Ab
    regression = RidgeRegression(A, b, standardize=True)
    regression.run_auto(min_alpha=-2, max_alpha=0, n_alphas=3)
    assert regression.alphas is not None
    assert regression.opt_alpha is not None
    assert regression.coeff is not None
    assert regression.psi is not None
    assert regression.errors is not None


def test_fit(si_111_Ab):
    """Test RidgeRegression._fit()."""
    alpha = 0.1
    A, b = si_111_Ab
    regression = RidgeRegression(A, b, standardize=True)
    regression._fit(alpha, debias=False)
    assert regression.coeff is not None


def test_predict(si_111_Ab):
    """Test RidgeRegression._predict()."""
    alpha = 0.1
    A, b = si_111_Ab
    regression = RidgeRegression(A, b, standardize=True)
    regression._fit(alpha, debias=False)
    prediction = regression._predict(A)
    np.testing.assert_array_equal(prediction.shape, b.shape)


def test_calc_error(si_111_Ab):
    """Test RidgeRegression._calc_error()."""
    alpha = 0.1
    A, b = si_111_Ab
    regression = RidgeRegression(A, b, standardize=True)
    regression._fit(alpha, debias=False)
    error = regression._calc_error(alpha)
    assert isinstance(error, float)


def test_calc_error_with_sklearn(si_111_Ab):
    """Test RidgeRegression._calc_error() compared with scikit-learn."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    alpha = 0.1
    A, b = si_111_Ab
    A = A[:1000]
    b = b[:1000]
    sc = StandardScaler()
    # use scikit-learn
    ridge_sk = Ridge(alpha=alpha, fit_intercept=False)
    loocv = LeaveOneOut()
    scaler = sc.fit(A)
    A_std = scaler.transform(A)
    errors = []
    for train_idx, test_idx in loocv.split(A_std):
        A_train, A_test = A_std[train_idx], A_std[test_idx]
        b_train, b_test = b[train_idx], b[test_idx]
        ridge_sk.fit(A_train, b_train)
        b_pred = ridge_sk.predict(A_test)
        error = np.mean((b_pred - b_test) ** 2)
        errors.append(error)
    mean_error_sk = sum(errors) / len(errors)
    # use our method
    regression = RidgeRegression(A, b, standardize=True)
    regression.run(alpha=alpha)
    mean_error = regression.errors[0]
    np.testing.assert_allclose(mean_error, mean_error_sk)


def test_calc_RTR_inv(si_111_Ab):
    """Test RidgeRegression._calc_RTR_inv()."""
    alpha = 0.1
    A, b = si_111_Ab
    regression = RidgeRegression(A, b, standardize=True)
    assert regression._R is not None
    assert regression._V is not None
    R = regression._R
    RTR_inv = regression._calc_RTR_inv(alpha)
    RTR_inv_np = np.linalg.inv(R.T @ R + alpha * np.eye(R.shape[1]))
    np.testing.assert_allclose(RTR_inv, RTR_inv_np)


def test_standardize_data(si_111_Ab):
    """Test standardize_data()."""
    A, _ = si_111_Ab
    std_A, scale = standardize_data(A)
    np.testing.assert_array_equal(std_A.shape, A.shape)
    np.testing.assert_array_equal(scale.shape[0], A.shape[1])
