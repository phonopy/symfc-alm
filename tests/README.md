# Tests

## How to run

```
pytest
```

## Data
### `FORCE_SETS_NaCl`
2x2x2 supercell of NaCl conventional unitcell:
```yaml
unit_cell:
  lattice:
  - [     5.640560000000000,     0.000000000000000,     0.000000000000000 ] # a
  - [     0.000000000000000,     5.640560000000000,     0.000000000000000 ] # b
  - [     0.000000000000000,     0.000000000000000,     5.640560000000000 ] # c
  points:
  - symbol: Na # 1
    coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  - symbol: Na # 2
    coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  - symbol: Na # 3
    coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  - symbol: Na # 4
    coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  - symbol: Cl # 5
    coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  - symbol: Cl # 6
    coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  - symbol: Cl # 7
    coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  - symbol: Cl # 8
    coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
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
