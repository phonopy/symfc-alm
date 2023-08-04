"""RidgeRegression module."""

import numpy as np


def ridge_regression(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    auto: bool,
    standardize: bool = True,
):
    """Conduct RidgeRegression.run() or RidgeRegression.run_auto() function.

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
    standardize: bool, optional
        When set to ``True``, standardize the input matrix A.

    """
    model = RidgeRegression(A, b, standardize=standardize)
    if auto:
        model.run_auto()
    else:
        model.run(alpha=alpha)
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

    def __init__(self, A: np.ndarray, b: np.ndarray, standardize: bool = True):
        """Initialize the RidgeRegression class.

        Parameters
        ----------
        A : ndarray
            Matrix A, derived from displacements.
            shape=(3 * num_atoms * ndata, num_fc)
        b : ndarray
            Vector b, derived from atomic forces.
            shape=(3 * num_atoms * ndata,)
        standardize: bool, optional
            When set to ``True``, standardize the input matrix A.

        """
        self._psi = None
        self._coeff = None
        self._alphas = None
        self._opt_alpha = None
        self._errors = None
        self.initialize(A, b, standardize)

    def initialize(self, A: np.ndarray, b: np.ndarray, standardize: bool = True):
        """Initialize the input values.

        Parameters
        ----------
        See docstring of RidgeRegression().__init__() for parameter descriptions.

        """
        if standardize:
            self._standardize = True
            A, self._scale = standardize_data(A)
        else:
            self._standardize = False
            self._scale = None
        self._A = A
        self._b = b
        U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
        self._V = Vt.T
        self._R = U @ np.diag(sigma)

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

    def run(self, alpha: float = 0.1, debias: bool = True):
        """Fit force constants using a specific hyperparameter.

        Parameters
        ----------
        alpha: float
            Hyperparameter for regularization terms.
        debias: bool
            When set to ``True``, correcting for bias from ridge regression estimators.

        """
        self._fit(alpha, debias)
        self._errors = np.array([self._calc_error()])
        self._psi = trans_prestandardize(self._coeff, self._scale, self._standardize)

    def run_auto(
        self,
        min_alpha: int = -6,
        max_alpha: int = 1,
        n_alphas: int = 3,
        debias: bool = True,
    ):
        """Fit force constants with an optimized hyperparameter.

        Parameters
        ----------
        min_alpha: int
            Minimum value of hyperparameter alpha on a logarithmic scale.
        max_alpha: int
            Maximum value of hyperparameter alpha on a logarithmic scale.
        n_alphas: int
            The number of divisions for hyperparameters.
        debias: bool
            When set to ``True``, correcting for bias from ridge regression estimators.

        """
        self._alphas = np.logspace(max_alpha, min_alpha, num=n_alphas)
        self._errors = np.zeros(len(self._alphas))
        for i, alpha in enumerate(self._alphas):
            self._fit(alpha, debias)
            self._errors[i] = self._calc_error()

        self._opt_alpha = self._alphas[np.argmin(self._errors)]
        self._fit(self._opt_alpha, debias)
        self._psi = trans_prestandardize(self._coeff, self._scale, self._standardize)

    def _fit(self, alpha: float, debias: bool):
        r"""Fit force constants.

        The regression coefficients for Ridge regression are calculated
        using singular value decomposition as follows:

        V(R^{T}R + \alpha I)^{-1}R^{T}b

        where R is R = U @ Σ, and U, Σ, V are obtained from the
        singular value decomposition of Matrix A.

        Formula implemented is based on this paper:

            Ref. 1: Wessel N. van Wieringen. https://arxiv.org/abs/1509.09169

        Parameters
        ----------
        alpha: float
            Hyperparameter for regularization terms.
        debias: bool
            When set to ``True``, correcting for bias from ridge regression estimators.
            Formula implemented is based on this paper:

                Ref. 2: Zhang et al. https://arxiv.org/abs/2009.08071

        """
        self._RTR_inv = self._calc_RTR_inv(alpha)
        coeff = self._V @ self._RTR_inv @ self._R.T @ self._b
        if debias:
            AtA_inv = self._V @ self._RTR_inv @ self._V.T
            self._coeff = coeff + alpha * AtA_inv @ coeff
        else:
            self._coeff = coeff

    def _predict(self, A: np.ndarray):
        """Generate predictions from fitted parameters.

        Parameters
        ----------
        A : ndarray
            Matrix A, derived from displacements.
            shape=(3 * num_atoms * ndata, num_fc)

        Returns
        -------
        ndarray
            Predicted values from fitted model.

        """
        return A @ self._coeff

    def _calc_error(self, alpha: float = None):
        r"""Analytically calculate leave-one-out cross validation (LOOCV) error.

        The LOOCV error is calculated as:

        E_{LOOCV} = \frac{1}{n}\sum_{j=1}^{n}\left[\frac{1}{1-\beta_{j}}(b_{j}
                                                   - \hat{b_{j}}) \right]^{2}

        where beta is the diagonal component of

        H = R(R^{T}R + \alpha I)^{-1}R^{T}

        Here, R is R = U @ Σ, and U, Σ are obtained from the
        singular value decomposition of Matrix A.

        Formula implemented is based on this paper:

            Ref. 1: Wessel N. van Wieringen. https://arxiv.org/abs/1509.09169

        Parameters
        ----------
        alpha: float
            Hyperparameter for regularization terms.

        Returns
        -------
        error : float
            The mean squared error calculated by LOOCV.

        """
        if alpha is not None:
            self._RTR_inv = self._calc_RTR_inv(alpha)
        b_pred = self._predict(self._A)
        H = self._R @ self._RTR_inv @ self._R.T
        n = H.shape[0]
        error = 0
        for i in range(n):
            error += ((self._b[i] - b_pred[i]) / (1 - H[i, i])) ** 2
        error /= n

        return error

    def _calc_RTR_inv(self, alpha: float):
        r"""Return the inverse of a specific matrix.

        The inverse is calculated for the following equation:

        (R^{T}R + \alpha I)^{-1}

        where R is R = U @ Σ, and U, Σ are obtained from the
        singular value decomposition (SVD) of Matrix A.

        Parameters
        ----------
        alpha: float
            Hyperparameter for the regularization term.

        Returns
        -------
        RTR_inv : np.ndarray
            The inverse of the matrix (R^T @ R + alpha * I).
        """
        RTR = self._R.T @ self._R
        RTR[np.diag_indices_from(RTR)] += alpha
        RTR_inv = np.linalg.inv(RTR)
        return RTR_inv


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
