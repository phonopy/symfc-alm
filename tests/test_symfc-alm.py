"""Tests of symfc-alm API."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from symfc_alm import CellDataset, DispForceDataset, LinearModel, SymfcAlm

cwd = Path(__file__).parent


def test_df_dataset(nacl_222_dataset: DispForceDataset):
    """Test reading displacements-forces dataset."""
    d = nacl_222_dataset.displacements
    f = nacl_222_dataset.forces
    np.testing.assert_array_equal(d.shape, (200, 64, 3))
    np.testing.assert_array_equal(f.shape, (200, 64, 3))
    np.testing.assert_allclose(d[0, 0], [-0.01669067, -0.02212396, 0.01148703])
    np.testing.assert_allclose(f[0, 0], [0.02407506, 0.06391442, -0.02732372])
    np.testing.assert_allclose(d[-1, -1], [-0.02179827, 0.00598846, -0.01972242])
    np.testing.assert_allclose(f[-1, -1], [0.06387808, -0.01690191, 0.04503784])


def test_cell_dataset(nacl_222_structure: CellDataset):
    """Test cell dataset."""
    cell = nacl_222_structure
    np.testing.assert_allclose(cell.lattice.diagonal(), [11.2811199999999996] * 3)
    np.testing.assert_allclose(cell.points[-1], [0.50, 0.50, 0.75])
    np.testing.assert_array_equal(cell.points.shape, (len(cell), 3))
    assert len(cell.numbers) == len(cell)


def test_run_fc2_nacl(
    nacl_222_dataset: DispForceDataset, nacl_222_structure: CellDataset
):
    """Test SymfcAlm.run() with NaCl fc2.

    Note1
    -----
    `phonopy_NaCl.yaml` can be used with `force_constants_NaCl.hdf5` to run phonopy.

    Note2
    -----
    Fc2 was written in hdf5 by:

       with h5py.File(cwd / "force_constants_NaCl.hdf5", "w") as w:
           w.create_dataset("force_constants", data=fcs[0], compression="gzip")


    """
    with SymfcAlm(nacl_222_dataset, nacl_222_structure, log_level=0) as sfa:
        sfa.run(maxorder=1)
    assert sfa._alm is None

    with h5py.File(cwd / "force_constants_NaCl.hdf5") as f:
        fc2 = f["force_constants"][:]

    np.testing.assert_allclose(sfa.force_constants[0], fc2)


@pytest.mark.big
def test_run_fc2_nacl_ridge(
    nacl_222_dataset: DispForceDataset, nacl_222_structure: CellDataset
):
    """Test SymfcAlm.run() with NaCl fc2 using ridge regression.

    Note1
    -----
    See docstring of test_run_fc2_nacl().

    Note2
    -----
    See docstring of test_run_fc2_nacl().

    """
    with SymfcAlm(nacl_222_dataset, nacl_222_structure, log_level=0) as sfa:
        sfa.run(maxorder=1, auto=False, linear_model=LinearModel(2))
    assert sfa._alm is None

    with h5py.File(cwd / "force_constants_NaCl.hdf5") as f:
        fc2 = f["force_constants"][:]

    np.testing.assert_allclose(sfa.force_constants[0], fc2, rtol=1e-04, atol=1e-06)


def test_run_fc2_fc3_si(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset
):
    """Test SymfcAlm.run() with Si fc2 and fc3 simultaneously.

    Note1
    -----
    `phono3py_Si111.yaml` can be used with `force_constants_Si111.hdf5` to run phonopy.

    Note2
    -----
    Fc2 was written in hdf5 by:

       with h5py.File(cwd / "fc2_Si111.hdf5", "w") as w:
           w.create_dataset("force_constants", data=fcs[0], compression="gzip")
       with h5py.File(cwd / "fc3_Si111.hdf5", "w") as w:
           w.create_dataset("fc3", data=fcs[1], compression="gzip")


    """
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        sfa.run(maxorder=2)
    assert sfa._alm is None
    with h5py.File(cwd / "fc2_Si111.hdf5") as f:
        fc2 = f["force_constants"][:]
    with h5py.File(cwd / "fc3_Si111.hdf5") as f:
        fc3 = f["fc3"][:]

    natom = len(si_111_structure)
    assert fc2.shape == (natom, natom, 3, 3)
    assert fc3.shape == (natom, natom, natom, 3, 3, 3)
    np.testing.assert_allclose(sfa.force_constants[0], fc2)
    np.testing.assert_allclose(sfa.force_constants[1], fc3)


def test_run_fc2_fc3_si_ridge(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset
):
    """Test SymfcAlm.run() with Si fc2 and fc3 simultaneously using ridge regression.

    Note1
    -----
    See docstring of test_run_fc2_fc3_si().

    Note2
    -----
    See docstring of test_run_fc2_fc3_si().

    """
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        sfa.run(maxorder=2, auto=False, linear_model=LinearModel(2))
    assert sfa._alm is None
    with h5py.File(cwd / "fc2_Si111.hdf5") as f:
        fc2 = f["force_constants"][:]
    with h5py.File(cwd / "fc3_Si111.hdf5") as f:
        fc3 = f["fc3"][:]

    natom = len(si_111_structure)
    assert fc2.shape == (natom, natom, 3, 3)
    assert fc3.shape == (natom, natom, natom, 3, 3, 3)
    np.testing.assert_allclose(sfa.force_constants[0], fc2, rtol=1e-05, atol=1e-08)
    np.testing.assert_allclose(sfa.force_constants[1], fc3, rtol=1e-05, atol=1e-08)


def test_get_matrix_elements_fc2_si(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset
):
    """Test SymfcAlm.get_matrix_elements() with Si fc2.

    * A.shape[0] = b.shape[0] = np.prod(displacements)
    * A.shape[1] roughly corresponds to the number of symmetrycally independent
      force constants elements.

    """
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=1)
    assert sfa._alm is None
    assert A.shape[0] == np.prod(si_111_dataset.displacements.shape)
    assert A.shape[1] == 4
    assert A.shape[0] == b.shape[0]


def test_get_matrix_elements_fc3_si(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset
):
    """Test SymfcAlm.get_matrix_elements() with Si fc3.

    * A.shape[0] = b.shape[0] = np.prod(displacements)
    * A.shape[1] roughly corresponds to the number of symmetrycally independent
      force constants elements.

    """
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=2, nbody=[0, 3])
    assert sfa._alm is None
    assert A.shape[0] == np.prod(si_111_dataset.displacements.shape)
    assert A.shape[1] == 13
    assert A.shape[0] == b.shape[0]


def test_get_matrix_elements_fc2_fc3_si(
    si_111_dataset: DispForceDataset, si_111_structure: CellDataset
):
    """Test SymfcAlm.get_matrix_elements() with Si fc2 and fc3 simultaneously.

    * A.shape[0] = b.shape[0] = np.prod(displacements)
    * A.shape[1] roughly corresponds to the number of symmetrycally independent
      force constants elements.

    """
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=2)
    assert sfa._alm is None
    assert A.shape[0] == np.prod(si_111_dataset.displacements.shape)
    assert A.shape[1] == 17  # 4 + 13
    assert A.shape[0] == b.shape[0]


def test_get_matrix_elements_fc2_nacl(
    nacl_222_dataset: DispForceDataset, nacl_222_structure: CellDataset
):
    """Test SymfcAlm.get_matrix_elements() with NaCl fc2.

    * A.shape[0] = b.shape[0] = np.prod(displacements)
    * A.shape[1] roughly corresponds to the number of symmetrycally independent
      force constants elements.

    """
    with SymfcAlm(nacl_222_dataset, nacl_222_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=1)
    assert sfa._alm is None
    assert A.shape[0] == np.prod(nacl_222_dataset.displacements.shape)
    assert A.shape[1] == 31
    assert A.shape[0] == b.shape[0]
