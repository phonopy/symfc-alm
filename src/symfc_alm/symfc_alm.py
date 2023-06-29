"""Symfc-alm module."""
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import numpy.typing as npt
from alm import ALM
import lzma
import pathlib
import io
import os


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
    """Symfc-alm API."""

    def __init__(self, dataset: DispForceDataset, cell: CellDataset):
        """Init method."""
        self._dataset = dataset
        self._cell = cell

    def run(self, maxorder: int = 2, nbody: Optional[npt.ArrayLike] = None):
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
        cell = self._cell
        with ALM(cell.lattice, cell.points, cell.numbers, verbose=True) as alm:
            alm.define(maxorder, nbody=nbody)
            alm.set_constraint()
            alm.displacements = self._dataset.displacements
            alm.forces = self._dataset.forces
            X, y = alm.get_matrix_elements()
            print(X.shape, y.shape)
