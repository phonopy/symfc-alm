"""Symfc-alm module."""
from __future__ import annotations

import io
import lzma
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from alm import ALM


class ALMNotInstanciatedError(RuntimeError):
    """Error of that ALM is not instanciated."""

    pass


class ALMAlreadyExistsError(RuntimeError):
    """Error of that ALM is already instanciated."""

    pass


class LinearModel(Enum):
    """Linear model used for fitting force constants."""

    LinearRegression = 1
    Ridgeregression = 2


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


def read_dataset(fp: Union[str, bytes, os.PathLike, io.IOBase]):
    """Read displacements-forces dataset.

    Parameters
     ----------
     fp : filename or stream
         filename or stream.

    """
    if isinstance(fp, io.IOBase):
        data = np.loadtxt(fp).reshape(-1, 64, 6)
    else:
        ext = pathlib.Path(fp).suffix
        if ext == ".xz":
            _io = lzma
        else:
            _io = io
        with _io.open(fp, "rb") as f:
            data = np.loadtxt(f).reshape(-1, 64, 6)
    displacements = data[:, :, :3]
    forces = data[:, :, 3:]
    return DispForceDataset(displacements, forces)


@dataclass
class DispForceDataset:
    """Displacements-forces dataset."""

    displacements: np.ndarray
    forces: np.ndarray

    def __init__(self, displacements: npt.ArrayLike, forces: npt.ArrayLike):
        """Init method."""
        self.displacements = np.array(displacements, dtype="double", order="C")
        self.forces = np.array(forces, dtype="double", order="C")


@dataclass
class CellDataset:
    """Crystal structure.

    lattice : ndarray
        Basis vectors. a, b, c are given as row vectors.
        shape=(3, 3), dtype='double'
    positions : ndarray
        Fractional coordinates of atomic points.
        shape=(num_atoms, 3), dtype='double'
    numbers : ndarray
        Atomic numbers.
        shape=(num_atoms,), dtype='intc'.

    """

    lattice: np.ndarray
    points: np.ndarray
    numbers: np.ndarray

    def __init__(
        self, lattice: npt.ArrayLike, points: npt.ArrayLike, numbers: npt.ArrayLike
    ):
        """Init method."""
        self.lattice = np.array(lattice, dtype="double", order="C")
        self.points = np.array(points, dtype="double", order="C")
        self.numbers = np.array(numbers, dtype="intc")
        if len(self.numbers) != len(self.points):
            raise RuntimeError("Shapes of numbers and points are inconsistent.")
        if self.points.shape[1] != 3:
            raise TypeError("Shape of second dimension of points has to be 3.")
        if self.lattice.shape != (3, 3):
            raise TypeError("Shape of lattice has to be (3, 3).")

    def __len__(self):
        """Return number of atoms."""
        return len(self.numbers)


class SymfcAlm:
    """Symfc-alm API.

    SymfcAlm.alm_new() has to be called after instantiated and
    SymfcAlm.alm_delete() has to be called after using. These can be handled by
    context manager, e.g.,

    with SymfcAlm(dataset, cell) as sfa:
        sfa.run()
    force_constants = sfa.force_constants

    """

    def __init__(
        self, dataset: DispForceDataset, cell: CellDataset, log_level: int = 0
    ):
        """Init method."""
        self._dataset = dataset
        self._cell = cell
        self._log_level = log_level
        self._alm: Optional[ALM] = None

    @property
    def force_constants(self) -> list[np.ndarray]:
        """Return force constants."""
        return self._force_constants

    def __enter__(self):
        """Create ALM instance."""
        self.alm_new()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Delete ALM instance."""
        self.alm_delete()

    def alm_new(self):
        """Create ALM instance.

        This create not only a python ALM instance but also the C++ ALM instance.
        So this has to be cleanly deleted by invoking self.alm_delete().

        """
        if self._alm is not None:
            raise ALMAlreadyExistsError("ALM is already instanticated.")

        cell = self._cell
        self._alm = ALM(
            cell.lattice, cell.points, cell.numbers, verbosity=self._log_level
        )
        self._alm.alm_new()

    def alm_delete(self):
        """Delete ALM instance."""
        if self._alm is not None:
            self._alm.alm_delete()
            self._alm = None

    def run(
        self,
        maxorder: int = 2,
        nbody: Optional[npt.ArrayLike] = None,
        linear_model: LinearModel = LinearModel.LinearRegression,
    ):
        """Compute force constants.

        Parameters
        ----------
        maxorder : int
            Upto (maxorder+1)-th order force constants are calculated.
        nbody : array_like of int
            For example, with maxorder=2,
            - nbody=[2, 3] : 2nd and 3rd order force constants, simultaneously
            - nbody=[0, 3] : only 3rd order force constants
            are computed. Default (None) gives
            ``[i + 2 for i in range(maxorder)]`` like the first example.

        """
        if self._alm is None:
            raise ALMNotInstanciatedError("ALM is not instanciated.")
        self.prepare(maxorder=maxorder, nbody=nbody)
        A, b = self._alm.get_matrix_elements()
        self.fit(A, b, linear_model=linear_model)
        self._force_constants = self._extract_fc_from_alm(self._alm, maxorder)

    def prepare(self, maxorder: int = 2, nbody: Optional[npt.ArrayLike] = None):
        """Prepare force constants calculation setting.

        Parameters
        ----------
        See docstring of SymfcAlm.run().

        """
        if self._alm is None:
            raise ALMNotInstanciatedError("ALM is not instanciated.")
        self._alm.define(maxorder, nbody=nbody)
        self._alm.set_constraint()
        self._alm.displacements = self._dataset.displacements
        self._alm.forces = self._dataset.forces

    def fit(
        self,
        A: np.ndarray,
        b: np.ndarray,
        linear_model: LinearModel = LinearModel.LinearRegression,
    ):
        """Fit force cosntants using matrices A and b.

        LinearModel.LinearRegression:
            psi = min_{psi} (A psi - b)

        """
        if self._alm is None:
            raise ALMNotInstanciatedError("ALM is not instanciated.")
        if linear_model is LinearModel.LinearRegression:
            psi = np.linalg.pinv(A) @ b
        else:
            raise RuntimeError("Unsupported linear model.")
        self._alm.set_fc(psi)

    def get_matrix_elements(
        self, maxorder: int = 2, nbody: Optional[npt.ArrayLike] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return matrix elements to compute force constants.

        Parameters
        ----------
        See docstring of SymfcAlm.run().

        Return
        ------
        tuple[A: np.ndarray, b: np.ndarray]
            Matrix A and vector b.
            When using least square fitting, force constants psi are computed by
                psi = A^~1.b

        """
        if self._alm is None:
            raise ALMNotInstanciatedError("ALM is not instanciated.")
        self._alm.define(maxorder, nbody=nbody)
        self._alm.set_constraint()
        self._alm.displacements = self._dataset.displacements
        self._alm.forces = self._dataset.forces
        A, b = self._alm.get_matrix_elements()

        return A, b

    def _extract_fc_from_alm(self, alm: ALM, maxorder):
        natom = len(self._cell)
        fcs = []
        for order in range(1, maxorder + 1):
            atom_list = np.arange(natom, dtype=int)
            fc_shape = (len(atom_list),) + (natom,) * order + (3,) * (order + 1)
            fc = np.zeros(fc_shape, dtype="double", order="C")
            for fc_elem, indices in zip(*alm.get_fc(order, mode="all")):
                v = indices // 3
                idx = np.where(atom_list == v[0])[0]
                if len(idx) > 0:
                    c = indices % 3
                    selection = (idx[0],) + tuple(v[1:]) + tuple(c)
                    fc[selection] = fc_elem

            fcs.append(fc)

        return fcs
