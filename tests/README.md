# Tests

## How to run

```
pytest
(When the "--runbig" option is set, it will also execute time-consuming tests.)
```

## Data

### `FORCE_SETS_NaCl`

2x2x2 supercell of NaCl conventional unitcell. 200 snapshots with 0.03 A random
displacements. Unit cell structure is

```yaml
unit_cell:
  lattice:
    - [5.640560000000000, 0.000000000000000, 0.000000000000000] # a
    - [0.000000000000000, 5.640560000000000, 0.000000000000000] # b
    - [0.000000000000000, 0.000000000000000, 5.640560000000000] # c
  points:
    - symbol: Na # 1
      coordinates: [0.000000000000000, 0.000000000000000, 0.000000000000000]
    - symbol: Na # 2
      coordinates: [0.000000000000000, 0.500000000000000, 0.500000000000000]
    - symbol: Na # 3
      coordinates: [0.500000000000000, 0.000000000000000, 0.500000000000000]
    - symbol: Na # 4
      coordinates: [0.500000000000000, 0.500000000000000, 0.000000000000000]
    - symbol: Cl # 5
      coordinates: [0.500000000000000, 0.500000000000000, 0.500000000000000]
    - symbol: Cl # 6
      coordinates: [0.500000000000000, 0.000000000000000, 0.000000000000000]
    - symbol: Cl # 7
      coordinates: [0.000000000000000, 0.500000000000000, 0.000000000000000]
    - symbol: Cl # 8
      coordinates: [0.000000000000000, 0.000000000000000, 0.500000000000000]
```

Forces are calculated by VASP code with the shifted 2x2x2
k-mesh and the following INCAR:

```
   PREC = Accurate
 IBRION = -1
 NELMIN = 5
  ENCUT = 500
  EDIFF = 1.000000e-08
 ISMEAR = 0
  SIGMA = 1.000000e-02
  IALGO = 38
  LREAL = .FALSE.
ADDGRID = .TRUE.
  LWAVE = .FALSE.
 LCHARG = .FALSE.
   NPAR = 4
   ISYM = 0
    GGA = PS
```

### `FORCE_SETS_Si111`

1x1x1 supercell of Si conventional unitcell. 1000 snapshots with random
displacements at 300K.

```yaml
unit_cell:
  lattice:
    - [5.433560030000000, 0.000000000000000, 0.000000000000000] # a
    - [0.000000000000000, 5.433560030000000, 0.000000000000000] # b
    - [0.000000000000000, 0.000000000000000, 5.433560030000000] # c
  points:
    - symbol: Si # 1
      coordinates: [0.875000000000000, 0.875000000000000, 0.875000000000000]
    - symbol: Si # 2
      coordinates: [0.875000000000000, 0.375000000000000, 0.375000000000000]
    - symbol: Si # 3
      coordinates: [0.375000000000000, 0.875000000000000, 0.375000000000000]
    - symbol: Si # 4
      coordinates: [0.375000000000000, 0.375000000000000, 0.875000000000000]
    - symbol: Si # 5
      coordinates: [0.125000000000000, 0.125000000000000, 0.125000000000000]
    - symbol: Si # 6
      coordinates: [0.125000000000000, 0.625000000000000, 0.625000000000000]
    - symbol: Si # 7
      coordinates: [0.625000000000000, 0.125000000000000, 0.625000000000000]
    - symbol: Si # 8
      coordinates: [0.625000000000000, 0.625000000000000, 0.125000000000000]
```

Forces are calculated by VASP code with the shifted 4x4x4
k-mesh and the following INCAR:

```
  PREC = Accurate
IBRION = -1
 EDIFF = 1e-8
NELMIN = 5
  NELM = 100
 ENCUT = 500
 IALGO = 38
ISMEAR = 0
   GGA = PS
 SIGMA = 0.01
 LREAL = False
lcharg = False
 lwave = False
```
