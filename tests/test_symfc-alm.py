"""Tests of symfc-alm API."""
import h5py
import numpy as np

from symfc_alm import CellDataset, DispForceDataset, SymfcAlm


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


def test_run_fc2(nacl_222_dataset: DispForceDataset, nacl_222_structure: CellDataset):
    """Test SymfcAlm.run() with NaCl fc2.

    Note1
    -----
    `phonopy_NaCl.yaml` can be used with `force_constants_NaCl.hdf5` to run phonopy.

    Note2
    -----
    Fc2 was written in hdf5 by:

       with h5py.File("force_constants_NaCl.hdf5", "w") as w:
           w.create_dataset("force_constants", data=fcs[0], compression="gzip")


    """
    sfa = SymfcAlm(nacl_222_dataset, nacl_222_structure, log_level=0)
    fcs = sfa.run(maxorder=1)

    with h5py.File("force_constants_NaCl.hdf5") as f:
        fc2 = f["force_constants"][:]

    np.testing.assert_allclose(fcs[0], fc2)
