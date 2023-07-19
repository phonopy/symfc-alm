"""Pytest conftest.py."""
from pathlib import Path

import numpy as np
import pytest

from symfc_alm import CellDataset, DispForceDataset, read_dataset

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
