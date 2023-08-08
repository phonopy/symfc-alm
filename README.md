# symfc-alm
Force constants calculator using ALM

Github: [https://github.com/phonopy/symfc-alm](https://github.com/phonopy/symfc-alm)

## Features

- Estimating force constants using the least squares method.

- Estimating force constants using the ridge regression method and analytically calculating leave-one-out cross validation (LOOCV) error.

## Usage

Below is an example of calculating the force constants up to the 3rd order in Si.

```python
import numpy as np
from symfc_alm import CellDataset, DispForceDataset, LinearModel, read_dataset, SymfcAlm

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
si_111_structure = CellDataset(lattice, points, numbers)
path = <path-to-symfc-alm>
si_111_dataset = read_dataset(os.path.join(path, "/FORCE_SETS_Si111.xz"))

with SymfcAlm(si_111_dataset, si_111_structure, log_level=0) as sfa:
		# When set to maxorder=2, it calculates the force constants up to the 3rd order.
		# When set to Linearmode(1), it uses the least squares method.
    sfa.run(maxorder=2, linear_model=LinearModel(1))
    fc2 = sfa.force_constants[0]
    fc3 = sfa.force_constants[1]
print(f"fc2: {fc2.shape}")
# -> fc2: (8, 8, 3, 3)
print(f"fc3: {fc3.shape}")
# -> fc3: (8, 8, 8, 3, 3, 3)
```

## Dependency

- numpy>=1.17.0
- alm

## License

symfc-alm is released under a BSD 3-clause license.

## Installation

```
conda create -n symfc-alm python>=3.8
conda activate symfc-alm
git clone git@github.com/phonopy/symfc-alm.git
cd symfc-alm
pip install -e .
```

## Development

The development of symfc-alm is managed on the `develop` branch of github symfc-alm repository.

- Github issues is the place to discuss about symfc-alm issues.
- Github pull request is the place to request merging source code.

### Formatting

Formatting rule is written in `pyproject.toml`.

### pre-commit

Pre-commit ([https://pre-commit.com/](https://pre-commit.com/)) is mainly used for applying the formatting rule automatically. Therefore, the use is strongly encouraged at or before git-commit. Pre-commit is set-up and used in the following way:

- Installed by `pip install pre-commit`, `conda install pre_commit` or see [https://pre-commit.com/#install](https://pre-commit.com/#install).
- pre-commit hook is installed by `pre-commit install`.
- pre-commit hook is run by `pre-commit run --all-files`.

Unless running pre-commit, [pre-commit.ci](http://pre-commit.ci/) may push the fix at PR by github action. In this case, the fix should be merged by the contributor's repository.

### VSCode setting

- Not strictly, but VSCode's `settings.json` may be written like

  ```json
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=88", "--ignore=E203,W503"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.pycodestyleEnabled": false,
  "python.linting.pydocstyleEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.sortImports.args": ["--profile", "black"],
  "[python]": {
      "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
  }
  ```

## Tests

Tests are written using pytest. To run tests, pytest has to be installed. The tests can be run by

```bash

% pytest

```
