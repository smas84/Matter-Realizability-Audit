# Matter Realizability Audit (MRA)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20934770.svg)](https://doi.org/10.5281/zenodo.20934770)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![SymPy 1.14](https://img.shields.io/badge/SymPy-1.14-green)
![License BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-lightgrey)

A symbolic diagnostic framework for assessing whether prescribed
perturbative spacetime geometries admit physically realizable matter
sectors in general relativity.

---

## Overview

This repository contains the reference implementation of the Matter
Realizability Audit (MRA) described in the accompanying manuscript:

> Salehizadeh, S., "Assessing Matter Realizability in Perturbative
> Cosmological Spacetimes," manuscript in preparation (2026).

The MRA is a hierarchical symbolic framework that separates geometric
validation from matter-sector realizability through benchmark
reconstruction, exact off-shell contracted second Bianchi identity
verification, conservation diagnostics, closure analysis, residual-ledger
construction, invariant decomposition, and realizability assessment.

All symbolic results reported in the accompanying manuscript are reproduced
directly by the reference implementation contained in this repository.

---

## Repository Contents

```
Matter-Realizability-Audit/
├── README.md                   # This file
├── LICENSE                     # BSD 3-Clause License
├── CITATION.cff                # Citation metadata (DOI-linked)
├── flrw_grand_audit_final.py   # Reference implementation of the MRA
├── requirements.txt            # Python package dependencies
└── run_audit.sh                # Shell script to run the audit
```

---

## Requirements

### Python version

Python 3.10 or later is required. The implementation was developed and
tested under Python 3.12.3.

### Package dependencies

All dependencies are listed in `requirements.txt`. The only required
package is SymPy:

```
sympy>=1.12
```

The implementation was developed and tested with SymPy 1.14.0. Earlier
versions of SymPy (1.12 or later) should be compatible but have not been
independently verified.

No additional packages (NumPy, SciPy, etc.) are required. The audit is
purely symbolic.

### Installation

```bash
pip install -r requirements.txt
```

---

## Running the Audit

### Full audit

```bash
python3 flrw_grand_audit_final.py
```

Or using the provided shell script:

```bash
bash run_audit.sh
```

To save output to a file:

```bash
bash run_audit.sh --save
```

This writes all output to `audit_output.txt` while also printing to the
terminal.

---

## Expected Runtime

Expected runtimes on a modern workstation (single core, Python 3.12,
SymPy 1.14):

| Audit Stage                         | Approximate Time |
|-------------------------------------|-----------------|
| Background geometry + Christoffel   | 30–60 s         |
| Benchmark validation (3 geometries) | 2–5 min         |
| Off-shell Bianchi verification      | 3–8 min         |
| Linearized Einstein tensor δG       | 5–15 min        |
| Matter conservation diagnostics     | 5–15 min        |
| Closure + residual ledger           | 5–10 min        |
| Invariant decomposition (q², Π²)    | 3–8 min         |
| **Total (full audit)**              | **~30–60 min**  |

Runtime depends on the host machine and SymPy version. The audit is
single-threaded.

---

## Expected Outputs

### Geometric validation (Levels 0–2)

```
[PASS] Minkowski benchmark: G_munu = 0
[PASS] Schwarzschild benchmark: G_munu = 0
[PASS] FLRW benchmark: standard Friedmann Einstein tensor
[PASS] Off-shell Bianchi identity: nabla_mu G^mu_nu = 0  (all components)
```

### Matter diagnostics (Levels 3–4)

```
Scalar closure:
  drho_sol = 2*G*M*rho0 / (r*a(t))
  dp_sol   = (6*G*M*p0 + 2*G*M*rho0) / (3*r*a(t))
Closure Jacobian det: nonzero  [non-degenerate]
```

### Invariant fingerprint (Level 4)

```
Pi^2 = G**2*M**2*(32*H**2*omega**2*r**2*a(t)**4*sin(theta)**2 + 6)
       / (r**6*a(t)**6)

Symbolic independence verification:
  dPi2/dv_r   = 0  [PASS]
  dPi2/dv_phi = 0  [PASS]
  dPi2/ddrho  = 0  [PASS]
  dPi2/ddp    = 0  [PASS]
  Pi^2 > 0        [PASS]  (strictly positive throughout perturbative domain)
```

### Realizability assessment (Level 5)

```
GEOMETRIC SECTOR:      Established   (independent of matter realization)
MATTER DIAGNOSTICS:    Established   (consistent diagnostic sequence)
RESIDUAL STRUCTURE:    Empirical     (non-vanishing post-closure residual)
INVARIANT q^2:         Empirical     (realization-dependent)
INVARIANT Pi^2:        Empirical     (nonzero, velocity-independent,
                                      realization-class robust)
GENERAL REALIZABILITY: Open Question
```

---

## Perturbative Geometry

The audit is applied to an axisymmetric Kerr-like frame-dragging
perturbation of a spatially flat FLRW background:

```
g_munu = g^(0)_munu + epsilon * h_munu
```

with nonzero perturbative components:

```
h_tt   = 2*G*M / (a(t)*r)
h_tphi = h_phit = 2*G*M*a(t)*omega*sin^2(theta)
```

All other components vanish. Here M is the characteristic mass scale,
omega parametrises the frame-dragging strength, and a(t) is the
cosmological scale factor.

---

## Reproducibility

All symbolic results in the manuscript — including the explicit form of
Π², the scalar closure solutions, and the independence verification
checks — are reproduced exactly by running the audit. To verify a
specific result, search the output for the corresponding diagnostic label
(e.g. `Pi2`, `drho_sol`, `C_G_full`).

---

## Citation

If you use this software in your research, please cite both the
accompanying manuscript and the software archive:

**Manuscript:**
```
Salehizadeh, S., "Assessing Matter Realizability in Perturbative
Cosmological Spacetimes," manuscript in preparation (2026).
```

**Software archive:**
```
Salehizadeh, S., Matter Realizability Audit (MRA) v1.0.0,
Zenodo (2026), DOI: 10.5281/zenodo.20934770
```

Citation metadata is also provided in `CITATION.cff`.

---

## License

This project is licensed under the BSD 3-Clause License.
See the `LICENSE` file for details.
