"""Pytest conftest.py."""
from pathlib import Path

import numpy as np
import pytest

from symfc_alm import CellDataset, DispForceDataset, SymfcAlm, read_dataset

cwd = Path(__file__).parent


def pytest_addoption(parser):
    """Add command option to pytest."""
    parser.addoption(
        "--runbig", action="store_true", default=False, help="run big tests"
    )


def pytest_configure(config):
    """Set up marker big."""
    config.addinivalue_line("markers", "big: mark test as big to run")


def pytest_collection_modifyitems(config, items):
    """Add mechanism to run with --runbig."""
    if config.getoption("--runbig"):
        # --runbig given in cli: do not skip slow tests
        return
    skip_big = pytest.mark.skip(reason="need --runbig option to run")
    for item in items:
        if "big" in item.keywords:
            item.add_marker(skip_big)


@pytest.fixture(scope="session")
def nacl_222_structure() -> CellDataset:
    """Return NaCl 2x2x2 supercell structure."""
    points = [
        [0.00, 0.00, 0.00],
        [0.50, 0.00, 0.00],
        [0.00, 0.50, 0.00],
        [0.50, 0.50, 0.00],
        [0.00, 0.00, 0.50],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.50, 0.50, 0.50],
        [0.00, 0.25, 0.25],
        [0.50, 0.25, 0.25],
        [0.00, 0.75, 0.25],
        [0.50, 0.75, 0.25],
        [0.00, 0.25, 0.75],
        [0.50, 0.25, 0.75],
        [0.00, 0.75, 0.75],
        [0.50, 0.75, 0.75],
        [0.25, 0.00, 0.25],
        [0.75, 0.00, 0.25],
        [0.25, 0.50, 0.25],
        [0.75, 0.50, 0.25],
        [0.25, 0.00, 0.75],
        [0.75, 0.00, 0.75],
        [0.25, 0.50, 0.75],
        [0.75, 0.50, 0.75],
        [0.25, 0.25, 0.00],
        [0.75, 0.25, 0.00],
        [0.25, 0.75, 0.00],
        [0.75, 0.75, 0.00],
        [0.25, 0.25, 0.50],
        [0.75, 0.25, 0.50],
        [0.25, 0.75, 0.50],
        [0.75, 0.75, 0.50],
        [0.25, 0.25, 0.25],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.25],
        [0.75, 0.75, 0.25],
        [0.25, 0.25, 0.75],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
        [0.75, 0.75, 0.75],
        [0.25, 0.00, 0.00],
        [0.75, 0.00, 0.00],
        [0.25, 0.50, 0.00],
        [0.75, 0.50, 0.00],
        [0.25, 0.00, 0.50],
        [0.75, 0.00, 0.50],
        [0.25, 0.50, 0.50],
        [0.75, 0.50, 0.50],
        [0.00, 0.25, 0.00],
        [0.50, 0.25, 0.00],
        [0.00, 0.75, 0.00],
        [0.50, 0.75, 0.00],
        [0.00, 0.25, 0.50],
        [0.50, 0.25, 0.50],
        [0.00, 0.75, 0.50],
        [0.50, 0.75, 0.50],
        [0.00, 0.00, 0.25],
        [0.50, 0.00, 0.25],
        [0.00, 0.50, 0.25],
        [0.50, 0.50, 0.25],
        [0.00, 0.00, 0.75],
        [0.50, 0.00, 0.75],
        [0.00, 0.50, 0.75],
        [0.50, 0.50, 0.75],
    ]
    lattice = np.eye(3) * 11.2811199999999996
    numbers = [11] * 32 + [17] * 32
    return CellDataset(lattice, points, numbers)


@pytest.fixture(scope="session")
def nacl_222_dataset() -> DispForceDataset:
    """Return NaCl 2x2x2 dataset."""
    return read_dataset(cwd / "FORCE_SETS_NaCl.xz")


@pytest.fixture(scope="session")
def si_111_structure() -> CellDataset:
    """Return Si 1x1x1 supercell structure."""
    lattice = np.eye(3) * 5.43356
    points = [
        [0.875, 0.875, 0.875],
        [0.875, 0.375, 0.375],
        [0.375, 0.875, 0.375],
        [0.375, 0.375, 0.875],
        [0.125, 0.125, 0.125],
        [0.125, 0.625, 0.625],
        [0.625, 0.125, 0.625],
        [0.625, 0.625, 0.125],
    ]
    numbers = [14] * 8
    return CellDataset(lattice, points, numbers)


@pytest.fixture(scope="session")
def si_111_dataset() -> DispForceDataset:
    """Return Si 1x1x1 dataset."""
    return read_dataset(cwd / "FORCE_SETS_Si111.xz")


@pytest.fixture(scope="session")
def si_111_Ab(si_111_dataset: DispForceDataset, si_111_structure: CellDataset):
    """Use Si fc3, generate matrix A from displacements and vector b from forces."""
    with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=2)
    return A, b


@pytest.fixture(scope="session")
def aln_332_structure() -> CellDataset:
    """Return AlN 3x3x2 supercell structure."""
    points = [
        [0.11111, 0.22222, 0.00047],
        [0.44444, 0.22222, 0.00047],
        [0.77778, 0.22222, 0.00047],
        [0.11111, 0.55556, 0.00047],
        [0.44444, 0.55556, 0.00047],
        [0.77778, 0.55556, 0.00047],
        [0.11111, 0.88889, 0.00047],
        [0.44444, 0.88889, 0.00047],
        [0.77778, 0.88889, 0.00047],
        [0.11111, 0.22222, 0.50047],
        [0.44444, 0.22222, 0.50047],
        [0.77778, 0.22222, 0.50047],
        [0.11111, 0.55556, 0.50047],
        [0.44444, 0.55556, 0.50047],
        [0.77778, 0.55556, 0.50047],
        [0.11111, 0.88889, 0.50047],
        [0.44444, 0.88889, 0.50047],
        [0.77778, 0.88889, 0.50047],
        [0.22222, 0.11111, 0.25047],
        [0.55556, 0.11111, 0.25047],
        [0.88889, 0.11111, 0.25047],
        [0.22222, 0.44444, 0.25047],
        [0.55556, 0.44444, 0.25047],
        [0.88889, 0.44444, 0.25047],
        [0.22222, 0.77778, 0.25047],
        [0.55556, 0.77778, 0.25047],
        [0.88889, 0.77778, 0.25047],
        [0.22222, 0.11111, 0.75047],
        [0.55556, 0.11111, 0.75047],
        [0.88889, 0.11111, 0.75047],
        [0.22222, 0.44444, 0.75047],
        [0.55556, 0.44444, 0.75047],
        [0.88889, 0.44444, 0.75047],
        [0.22222, 0.77778, 0.75047],
        [0.55556, 0.77778, 0.75047],
        [0.88889, 0.77778, 0.75047],
        [0.11111, 0.22222, 0.30953],
        [0.44444, 0.22222, 0.30953],
        [0.77778, 0.22222, 0.30953],
        [0.11111, 0.55556, 0.30953],
        [0.44444, 0.55556, 0.30953],
        [0.77778, 0.55556, 0.30953],
        [0.11111, 0.88889, 0.30953],
        [0.44444, 0.88889, 0.30953],
        [0.77778, 0.88889, 0.30953],
        [0.11111, 0.22222, 0.80953],
        [0.44444, 0.22222, 0.80953],
        [0.77778, 0.22222, 0.80953],
        [0.11111, 0.55556, 0.80953],
        [0.44444, 0.55556, 0.80953],
        [0.77778, 0.55556, 0.80953],
        [0.11111, 0.88889, 0.80953],
        [0.44444, 0.88889, 0.80953],
        [0.77778, 0.88889, 0.80953],
        [0.22222, 0.11111, 0.05953],
        [0.55556, 0.11111, 0.05953],
        [0.88889, 0.11111, 0.05953],
        [0.22222, 0.44444, 0.05953],
        [0.55556, 0.44444, 0.05953],
        [0.88889, 0.44444, 0.05953],
        [0.22222, 0.77778, 0.05953],
        [0.55556, 0.77778, 0.05953],
        [0.88889, 0.77778, 0.05953],
        [0.22222, 0.11111, 0.55953],
        [0.55556, 0.11111, 0.55953],
        [0.88889, 0.11111, 0.55953],
        [0.22222, 0.44444, 0.55953],
        [0.55556, 0.44444, 0.55953],
        [0.88889, 0.44444, 0.55953],
        [0.22222, 0.77778, 0.55953],
        [0.55556, 0.77778, 0.55953],
        [0.88889, 0.77778, 0.55953],
    ]
    lattice = [
        [9.333, 0.0, 0.0],
        [-4.6665, 8.08261509, 0.0],
        [0.0, 0.0, 9.956],
    ]
    numbers = [13] * 36 + [7] * 36
    return CellDataset(lattice, points, numbers)


@pytest.fixture(scope="session")
def aln_332_dataset() -> DispForceDataset:
    """Return AlN 3x3x2 dataset.

    Note1
    -----
    Return 'DispForceDataset' of AlN is rank deficient data.

    """
    aln_dataset = read_dataset(cwd / "FORCE_SETS_AlN.xz")
    aln_dataset.displacements = aln_dataset.displacements.reshape(-1, 72, 3)
    aln_dataset.forces = aln_dataset.forces.reshape(-1, 72, 3)
    return aln_dataset


@pytest.fixture(scope="session")
def aln_332_Ab(aln_332_dataset: DispForceDataset, aln_332_structure: CellDataset):
    """Use AlN fc3, generate matrix A from displacements and vector b from forces."""
    with SymfcAlm(aln_332_dataset, aln_332_structure, log_level=0) as sfa:
        A, b = sfa.get_matrix_elements(maxorder=2)
    return A, b
