"""RidgeRegression module."""

import numpy as np


def ridge_regression(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    auto: bool,
):
    """Conduct RidgeRegression.run().

    Parameters
    ----------
    A : ndarray
        Matrix A, derived from displacements.
        shape=(3 * num_atoms * ndata, num_fc)
    b : ndarray
        Vector b, derived from atomic forces.
        shape=(3 * num_atoms * ndata,)
    alpha: float
        Hyperparameter for regularization terms.
    auto: bool
        When set to ``True``, the optimal alpha is automatically determined.

    """
    model = RidgeRegression()
    if auto:
        model.run_auto(A, b)
    else:
        model.run(A, b, alpha=alpha)
    psi = model.psi
    return psi


class RidgeRegression:
    r"""Ridge Regression model using leave-one-out cross validation.

    In the Ridge Regression technique, one solves the following equation:

    \tilde{\Phi} = arg\underset{\Phi}{min}\lVert b-A\Phi\rVert^{2}_{2}
                   +\alpha\lVert\Phi\rVert^{2}_{2}

    where the matrix A is derived from displacements, b is derived from atomic forces,
    and alpha is a hyperparameter.

    Attributes
    ----------
    psi : ndarray
        The matrix of irreducible set of force constants.
        shape=(num_fc,)
    coeff : ndarray
        The matrix of regression coefficients.
        shape=(num_fc,)
    alphas : ndarray
        The array of hyperparameters used for ridge regression.
        shape=(num_alpha,)
    opt_alpha: float
        Hyperparameter that results in minimal errors.
    errors: ndarray
        Array of errors corresponding to each hyperparameter in 'alphas'.
        shape=(num_error,)

    """

    def __init__(self):
        """Initialize the RidgeRegression class."""
        self._psi = None
        self._coeff = None
        self._alphas = None
        self._opt_alpha = None
        self._errors = None

    @property
    def psi(self) -> np.ndarray:
        """Return psi matrix."""
        return self._psi

    @property
    def coeff(self) -> np.ndarray:
        """Return regression coefficients."""
        return self._coeff

    @property
    def alphas(self) -> np.ndarray:
        """Return the list of alpha."""
        return self._alphas

    @property
    def opt_alpha(self) -> float:
        """Return the optimal alpha value."""
        return self._opt_alpha

    @property
    def errors(self) -> np.ndarray:
        """Return the list of errors."""
        return self._errors

    def run(
        self, A: np.ndarray, b: np.ndarray, alpha: float = 0.1, standardize: bool = True
    ):
        """Fit force constants using a specific hyperparameter.

        Parameters
        ----------
        A : ndarray
            Matrix A, derived from displacements.
            shape=(3 * num_atoms * ndata, num_fc)
        b : ndarray
            Vector b, derived from atomic forces.
            shape=(3 * num_atoms * ndata,)
        alpha: float
            Hyperparameter for regularization terms.
        standardize: bool, optional
            When set to ``True``, standardize the input matrix A.

        """
        if standardize:
            A, scale = standardize_data(A)

        self._fit(A, b, alpha)
        self._errors = np.array([self._calc_error(A, b, alpha)])
        self._psi = trans_prestandardize(self._coeff, scale, standardize)

    def run_auto(
        self,
        A: np.ndarray,
        b: np.ndarray,
        standardize: bool = True,
        min_alpha: int = -6,
        max_alpha: int = 1,
        n_alphas: int = 100,
    ):
        """Fit force constants with an optimized hyperparameter.

        Parameters
        ----------
        A : See docstring of RidgeRegression.run().
        b : See docstring of RidgeRegression.run().
        standardize: See docstring of RidgeRegression.run().
        min_alpha: int
            Minimum value of hyperparameter alpha on a logarithmic scale.
        max_alpha: int
            Maximum value of hyperparameter alpha on a logarithmic scale.
        n_alphas: int
            The number of divisions for hyperparameters.

        """
        if standardize:
            A, scale = standardize_data(A)

        self._alphas = np.logspace(max_alpha, min_alpha, num=n_alphas)
        self._errors = np.zeros(len(self._alphas))
        for i, alpha in enumerate(self._alphas):
            self._fit(A, b, alpha)
            self._errors[i] = self._calc_error(A, b, alpha)

        self._opt_alpha = self._alphas[np.argmin(self._errors)]
        self._fit(A, b, self._opt_alpha)
        self._psi = trans_prestandardize(self._coeff, scale, standardize)

    def _fit(self, A: np.ndarray, b: np.ndarray, alpha: float):
        """Fit force constants.

        Parameters
        ----------
        See docstring of RidgeRegression.run() for parameter descriptions.

        """
        self._coeff = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)

    def _predict(self, A: np.ndarray):
        """Generate predictions from fitted parameters.

        Parameters
        ----------
        A : See docstring of RidgeRegression.run().

        Returns
        -------
        ndarray
            Predicted values from fitted model.

        """
        return A @ self._coeff

    def _calc_error(self, A: np.ndarray, b: np.ndarray, alpha: float):
        r"""Analytically calculate leave-one-out cross validation (LOOCV) error.

        The LOOCV error is calculated as:

        E_{LOOCV} = \frac{1}{n}\sum_{j=1}^{n}\left[\frac{1}{1-\beta_{j}}(b_{j}
                                                   - \hat{b_{j}}) \right]^{2}

        where beta is the diagonal component of

        H = A(A^{T}A + \alpha I)^{-1}A^{T}

        Parameters
        ----------
        See docstring of RidgeRegression.run() for parameter descriptions.

        Returns
        -------
        error : float
            The mean squared error calculated by LOOCV.

        """
        b_pred = self._predict(A)
        H = A @ np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1])) @ A.T
        beta = np.diag(H)
        error = np.mean(((b - b_pred) / (1 - beta)) ** 2)

        return error


def standardize_data(A: np.ndarray):
    """Standardize the input data.

    Parameters
    ----------
    A : ndarray
        The data matrix.

    Returns
    -------
    std : ndarray
        Standardized data matrix.
    scale: ndarray
        Scale factors (standard deviations).

    """
    mean = A.mean(axis=0, keepdims=True)
    scale = A.std(axis=0, keepdims=True)
    scale = np.where(scale < 10 * np.finfo(scale.dtype).eps, 1.0, scale)
    std = (A - mean) / scale
    return std, scale[0]


def trans_prestandardize(coeff: np.ndarray, scale: np.ndarray, standardize: bool):
    """Return the input regression coefficients to its pre-standardized form.

    Parameters
    ----------
    coeff : ndarray
        The matrix of regression coefficients.
        shape=(num_fc,)
    scale: ndarray
        Scale factors (standard deviations).
    standardize: bool, optional
        When set to ``True``, pre-standardize the input coeff.

    Returns
    -------
    psi : ndarray
        When set to standardize is ``True``, return the pre-standardized input coeff.

    """
    if standardize:
        psi = np.true_divide(coeff, scale)
    else:
        psi = coeff
    return psi
