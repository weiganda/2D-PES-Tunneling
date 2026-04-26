# 2D PES & WKB Tunnelling for Double Proton Transfer in G–C

A computational toolkit for constructing two-dimensional potential energy surfaces and computing quantum-mechanical tunnelling rates for double proton transfer in guanine–cytosine (G–C) base pairs.

## Background

Spontaneous tautomerisation of Watson–Crick base pairs via double proton transfer has been proposed as a source of point mutations in DNA. In the G–C pair, two protons can transfer concertedly or sequentially between hydrogen-bonding partners, converting the canonical form into a rare tautomeric form that can mispair during replication.

This toolkit maps the energetics of that process onto a 2D potential energy surface parameterised by the two donor–proton distances, then computes semiclassical (WKB) tunnelling corrections and thermal rate constants along both the minimum energy path (MEP) and the variationally optimised minimal action path (MAP).

## Programs

### `generate_2d_pes.py` — PES Construction

Automates the generation, execution, and post-processing of constrained geometry optimisations on a 2D grid of donor–proton distances.

**Reaction coordinates:**

| Coordinate | Proton | Donor | Acceptor | Description |
|---|---|---|---|---|
| d₁ | H5 (index 5) | N22 (index 22) | N20 (index 20) | Amino → ring nitrogen |
| d₂ | H4 (index 4) | N21 (index 21) | O28 (index 28) | Ring nitrogen → carbonyl oxygen |

**Subcommands:**

| Command | Purpose |
|---|---|
| `generate` | Create ORCA input files and SLURM submission scripts for the full grid |
| `collect` | Parse ORCA outputs, extract energies, convert to relative eV, clean scratch files |
| `plot` | Generate a publication-quality contour plot of the 2D PES |
| `restart` | Identify failed/incomplete grid points and write resubmission scripts |

**Computational details:**
- DFT level: B3LYP/def2-TZVP with RIJCOSX acceleration and DefGrid3
- Grid: 20 × 20 = 400 constrained optimisations (d₁, d₂ ∈ [0.90, 2.00] Å)
- Each grid point fixes d₁ and d₂ while relaxing all other internal coordinates

### `tunneling_map.py` — WKB Tunnelling & Rate Constants

Loads the 2D PES and computes tunnelling-corrected reaction rates via three approaches.

**Stages:**

1. **Surface preparation** — Gaussian smoothing + bicubic spline interpolation
2. **Critical-point analysis** — Locate minima and saddle point on the interpolated surface
3. **Path optimisation**
   - *MEP* via the simplified string method (perpendicular gradient projection + equal arc-length redistribution)
   - *MAP* via free-node variational minimisation of the WKB action integral, with self-consistent barrier iteration
4. **Rate calculation** — Boltzmann-weighted integration of WKB transmission T(E) to obtain κ(T), then tunnelling-corrected Eyring rate constants

**Key features:**
- Forward and reverse MAP paths are optimised independently
- MAP barrier is determined self-consistently (no MEP dependence)
- MEP geometry seeds the MAP optimiser but does not constrain it
- Optional multiprocessing for the energy integration and temperature sweep

## Requirements

- Python ≥ 3.6
- NumPy
- SciPy
- Matplotlib
- pandas
- [ORCA](https://www.faccts.de/orca/) ≥ 6.0 (for the quantum chemistry calculations, run on a cluster)

## Quick Start

### 1. Generate the 2D PES grid

Place your starting geometry as `canon.xyz` in the working directory, then:

```bash
python generate_2d_pes.py generate
```

This creates the `2d_pes_grid/` directory containing 400 ORCA input files and SLURM submission scripts.

### 2. Run the calculations

```bash
cd 2d_pes_grid
sbatch submit_array.sh          # array job (preferred)
# — or —
bash submit_all_individual.sh   # one sbatch per point
```

### 3. Collect energies

```bash
cd ..
python generate_2d_pes.py collect
```

### 4. Plot the PES

```bash
python generate_2d_pes.py plot
```

### 5. Compute tunnelling rates

```bash
python tunneling_map.py --grid-dir 2d_pes_grid -T 298.15
```

For parallel execution:

```bash
python tunneling_map.py --grid-dir 2d_pes_grid -T 298.15 --n-workers 8
```

### 6. Handle failed jobs (if needed)

```bash
python generate_2d_pes.py restart
cd 2d_pes_grid
sbatch resubmit_array.sh
```

## Configuration

### `generate_2d_pes.py`

All configuration is set via module-level constants at the top of the script:

| Parameter | Default | Description |
|---|---|---|
| `H1_INDEX`, `DONOR1_INDEX` | 5, 22 | Atom indices for the first proton transfer (H5–N22) |
| `H2_INDEX`, `DONOR2_INDEX` | 4, 21 | Atom indices for the second proton transfer (H4–N21) |
| `D1_MIN`, `D1_MAX` | 0.90, 2.00 | Scan range for d₁ (Å) |
| `D2_MIN`, `D2_MAX` | 0.90, 2.00 | Scan range for d₂ (Å) |
| `N_POINTS` | 20 | Grid points per axis |
| `METHOD` | B3LYP def2-TZVP ... | ORCA method line |
| `NPROCS` | 8 | MPI processes per job |
| `MAXCORE` | 3700 | Memory per core (MB) |
| `PARTITION`, `ACCOUNT` | — | SLURM scheduler settings |
| `WALLTIME` | 08:00:00 | Wall-clock limit per job |

### `tunneling_map.py`

All parameters are set via command-line arguments:

| Flag | Default | Description |
|---|---|---|
| `--grid-dir` | *(required)* | Directory with `.npy` files from `collect` |
| `-T` | 298.15 | Reporting temperature (K) |
| `--smooth` | 0.3 | Gaussian smoothing σ (grid-point units; 0 = none) |
| `--n-inst-nodes` | 24 | Free interior nodes in the MAP path |
| `--mass-factor` | 1.0 | Effective mass multiplier (1.0 = proton, 2.0 = deuterium) |
| `--map-scf-tol` | 1e-4 | Convergence tolerance for MAP barrier SCF (eV) |
| `--map-scf-maxiter` | 10 | Maximum MAP barrier SCF iterations |
| `--n-workers` | 1 | Parallel workers (1 = serial) |

## Output Files

### From `generate_2d_pes.py`

| File | Stage | Description |
|---|---|---|
| `2d_pes_grid/point_XX_YY.inp` | `generate` | ORCA input for grid point (XX, YY) |
| `2d_pes_grid/submit_array.sh` | `generate` | SLURM array job script |
| `2d_pes_grid/d1_values.npy` | `generate` | 1-D array of d₁ scan values |
| `2d_pes_grid/d2_values.npy` | `generate` | 1-D array of d₂ scan values |
| `2d_pes_grid/energies_eV.npy` | `collect` | 2-D array of relative energies (eV) |
| `2d_pes_grid/2d_pes.png` | `plot` | Contour plot of the 2D PES |

### From `tunneling_map.py`

| File | Description |
|---|---|
| `2d_mep_analysis.png` | MEP overlaid on the PES contour + energy profile |
| `2d_mep_vs_MAP.png` | MEP and both MAP paths on the PES |
| `2d_energy_mep_vs_MAP.png` | Energy profiles along MEP and MAP paths |
| `2d_WKB_Tunnelling_forward.png` | T(E) and Boltzmann-weighted T(E) for forward reaction |
| `2d_WKB_Tunnelling_reverse.png` | Same for reverse reaction |
| `2d_action_difference_fwd.png` | WKB action comparison: MEP vs MAP (forward) |
| `2d_action_difference_rev.png` | Same for reverse |
| `2d_WKB_eyring_rates_fwd.png` | Arrhenius plot: forward MEP vs MAP rates |
| `2d_WKB_eyring_rates_rev.png` | Same for reverse |
| `2d_WKB_rates_vs_temperature.csv` | Full numerical data (κ, k for all temperatures) |

## Adapting to Other Systems

This toolkit is not specific to the G–C base pair. To apply it to any two-coordinate proton transfer (or other reaction):

1. Replace `canon.xyz` with your starting geometry.
2. Update atom indices (`H1_INDEX`, `DONOR1_INDEX`, etc.) to match your system. ORCA uses 0-based indexing.
3. Adjust the scan ranges (`D1_MIN`/`D1_MAX`, `D2_MIN`/`D2_MAX`) to span the relevant bond distances.
4. Update the reference-geometry markers in the `plot_pes()` function.
5. Modify SLURM settings for your cluster.
6. Adjust `--mass-factor` if the tunnelling particle is not a proton (e.g., 2.0 for deuterium).

## Theory Notes

**WKB action integral:**
The semiclassical action along a tunnelling path is S = ∫ √(2m[V(s) − E] / ħ²) ds, integrated over the classically forbidden region where V(s) > E. The transmission coefficient is T(E) ≈ exp(−2S).

**Tunnelling correction factor:**
κ(T) = β · exp(βΔV) · ∫ T(E) · exp(−βE) dE, where β = 1/k_BT and ΔV is the classical barrier height. κ ≥ 1 by construction.

**Eyring rate constant:**
k(T) = κ · (k_BT / h) · exp(−ΔV / k_BT).

**MEP vs MAP:**
The MEP follows the valley floor of the PES — it is the path of steepest descent from the saddle point. The MAP minimises the WKB action and may deviate from the MEP by "corner-cutting" through higher-potential but narrower barrier regions, resulting in higher transmission. The MAP is the physically relevant tunnelling path.


## Citation

If you use this code in your research, please cite:

**Manuscript in Progress**

> **[Author(s)], "[Paper Title]," *[Journal]*, [Volume], [Pages] ([Year]).** DOI: [DOI]

```bibtex
@article{YourCiteKey,
  author  = {},
  title   = {},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {},
  doi     = {}
}
```

*Manuscript in preparation (2026). This section will be updated with full citation details upon publication.*
