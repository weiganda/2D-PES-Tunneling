#!/usr/bin/env python

"""
2D PES Grid Generator for G-C Double Proton Transfer
=====================================================

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 Code implementation and structure assisted by Claude (Anthropic),
                 April 2026.
LAST MODIFIED:   April 24, 2026


Overview
--------
This script automates the construction, execution, and analysis of a
two-dimensional potential energy surface (2D PES) for the concerted
double proton transfer in a guanine–cytosine (G–C) base pair.

The two reaction coordinates are the donor–proton bond distances:

  d₁ = d(H5–N22)   Proton H5 transferring from donor N22 → acceptor N20
  d₂ = d(H4–N21)   Proton H4 transferring from donor N21 → acceptor O28

Each grid point corresponds to a constrained geometry optimization at a
fixed (d₁, d₂) pair.  All remaining internal coordinates are relaxed by
ORCA at the B3LYP/def2-TZVP level with RIJCOSX acceleration.


Workflow
--------
The script supports four subcommands, intended to be run in order:

  1. generate  — Build ORCA input files for every grid point and write
                 SLURM submission scripts (array job + individual fallback).
  2. (run)     — Submit the jobs on your cluster (done manually).
  3. collect   — Parse completed ORCA outputs, extract final energies,
                 convert to relative eV, and save as a NumPy array.
                 Also cleans up scratch files.
  4. plot      — Load the saved energy array and produce a publication-
                 quality contour plot of the 2D PES.
  5. restart   — Identify grid points that failed or never finished and
                 generate resubmission scripts for only those points.


Grid Layout
-----------
Both d₁ and d₂ are scanned from 0.90 Å to 2.00 Å on a uniform
N_POINTS × N_POINTS grid (default 20 × 20 = 400 single-point
constrained optimizations).

The lower-left corner (short d₁, short d₂) corresponds to the
canonical Watson–Crick geometry; the upper-right corner (long d₁,
long d₂) corresponds to the rare tautomeric form produced by double
proton transfer.


File Structure
--------------
After running ``generate``, the output directory (default ``2d_pes_grid/``)
will contain::

    2d_pes_grid/
    ├── canon.xyz                  # Starting geometry (copied in)
    ├── d1_values.npy              # 1-D array of d₁ scan values
    ├── d2_values.npy              # 1-D array of d₂ scan values
    ├── point_00_00.inp            # ORCA input for grid point (0,0)
    ├── point_00_01.inp            # ...
    ├── ...
    ├── point_19_19.inp            # ORCA input for grid point (19,19)
    ├── submit_array.sh            # SLURM array job script
    └── submit_all_individual.sh   # Alternative: one sbatch per point

After ``collect``::

    2d_pes_grid/
    ├── point_00_00.out            # ORCA outputs (kept for reference)
    ├── ...
    ├── d1_values.npy
    ├── d2_values.npy
    ├── energies_eV.npy            # 2-D array of relative energies (eV)
    └── 2d_pes.png                 # (after ``plot``)


Dependencies
------------
  - Python 3.6+
  - NumPy
  - Matplotlib (for plotting only)
  - ORCA 6.x (run on the cluster; not needed locally)


Usage
-----
::

    python generate_2d_pes.py generate   # Create ORCA inputs + SLURM scripts
    python generate_2d_pes.py collect    # Parse outputs → energies_eV.npy
    python generate_2d_pes.py plot       # Contour plot of the 2D PES
    python generate_2d_pes.py restart    # Resubmit failed/incomplete points


Adapting to Other Systems
-------------------------
To apply this script to a different proton-transfer system:

  1. Replace ``canon.xyz`` with your starting geometry.
  2. Update the atom indices (H1_INDEX, DONOR1_INDEX, H2_INDEX,
     DONOR2_INDEX) to match your transferring protons and their donors.
     Remember: ORCA uses 0-based atom indexing.
  3. Adjust D1_MIN/D1_MAX and D2_MIN/D2_MAX to span the relevant
     bond-distance range for your reaction coordinates.
  4. Update the starred marker positions in ``plot_pes()`` to reflect
     the canonical and tautomeric geometries of your system.
  5. Modify SLURM settings (PARTITION, ACCOUNT, WALLTIME, etc.) to
     match your cluster environment.

"""

import numpy as np
import os
import matplotlib.pyplot as plt


# ================================================================
# SYSTEM CONFIGURATION
# ================================================================
# All atom indices follow ORCA's 0-based numbering (i.e., the first
# atom in the XYZ file is index 0).

# --- Reaction coordinate 1: H5 transfers from N22 (donor) to N20 (acceptor) ---
H1_INDEX = 5          # Atom index of the transferring proton (H5)
DONOR1_INDEX = 22     # Atom index of the proton donor (N22)
# Note: the acceptor (N20, index 20) is not constrained; only the
# donor–proton distance is fixed at each grid point.

# --- Reaction coordinate 2: H4 transfers from N21 (donor) to O28 (acceptor) ---
H2_INDEX = 4          # Atom index of the transferring proton (H4)
DONOR2_INDEX = 21     # Atom index of the proton donor (N21)

# --- Grid parameters ---
D1_MIN, D1_MAX = 0.90, 2.00   # Scan range for d₁ = d(H5–N22) in Angstroms
D2_MIN, D2_MAX = 0.90, 2.00   # Scan range for d₂ = d(H4–N21) in Angstroms
N_POINTS = 20                 # Number of points per axis → 20×20 = 400 jobs

# --- ORCA calculation settings ---
# B3LYP/def2-TZVP with RIJCOSX (RI-J + COSX for Coulomb/exchange integrals),
# def2/J auxiliary basis, constrained optimization (Opt), tight SCF convergence,
# and fine DFT integration grid (DefGrid3).
METHOD = "B3LYP def2-TZVP RIJCOSX def2/J Opt TightSCF DefGrid3"
NPROCS = 8            # Number of MPI processes per job
MAXCORE = 3700        # Memory per core in MB (~30 GB total / 8 cores)
XYZ_FILE = "canon.xyz"  # Starting geometry (canonical Watson–Crick G–C)

# --- Cluster / SLURM settings ---
ORCA_PATH = "/cluster/software/common/orca/v6.1.0/orca_6_1_0_linux_x86-64_shared_openmpi418/orca"
PARTITION = "general,rulisp-lab"
ACCOUNT = "rulisp-lab"
WALLTIME = "08:00:00"  # Wall-clock limit per job

# --- Output directory ---
GRID_DIR = "2d_pes_grid"


# ================================================================
# GENERATE ORCA INPUTS
# ================================================================

def generate_orca_input(d1, d2, i, j):
    """
    Write a single ORCA input file for grid point (i, j).

    The input constrains two bond distances during geometry optimization:
      - d(H5–N22) = d1
      - d(H4–N21) = d2

    All other internal coordinates are free to relax.

    Parameters
    ----------
    d1 : float
        Constrained distance for H5–N22 (Angstroms).
    d2 : float
        Constrained distance for H4–N21 (Angstroms).
    i : int
        Row index on the grid (corresponds to d1).
    j : int
        Column index on the grid (corresponds to d2).

    Returns
    -------
    str
        Path to the generated input file.
    """

    filename = f"{GRID_DIR}/point_{i:02d}_{j:02d}.inp"

    # ORCA constraint syntax: {B atom1 atom2 distance C}
    #   B = bond constraint
    #   C = constrain (freeze) this coordinate during optimization
    content = f"""! {METHOD}

%geom
  Constraints
    {{B {H1_INDEX} {DONOR1_INDEX} {d1:.4f} C}}
    {{B {H2_INDEX} {DONOR2_INDEX} {d2:.4f} C}}
  end
end

%pal
  nprocs {NPROCS}
end

%maxcore {MAXCORE}

* xyzfile 0 1 {XYZ_FILE}
"""
    with open(filename, 'w') as f:
        f.write(content)

    return filename


def generate_all_inputs():
    """
    Generate ORCA input files and SLURM submission scripts for the
    entire 2D grid.

    Creates:
      - One ``.inp`` file per grid point (N_POINTS² total).
      - ``d1_values.npy`` / ``d2_values.npy``: the 1-D arrays of scan
        values, saved for later use by ``collect`` and ``plot``.
      - ``submit_array.sh``: a SLURM array job that dispatches all grid
        points in a single ``sbatch`` call.
      - ``submit_all_individual.sh``: a fallback script that submits
        each point as a separate job (useful if array jobs are disabled
        or if you need finer control).
    """

    os.makedirs(GRID_DIR, exist_ok=True)

    # Build uniformly spaced scan arrays
    d1_values = np.linspace(D1_MIN, D1_MAX, N_POINTS)
    d2_values = np.linspace(D2_MIN, D2_MAX, N_POINTS)

    # Persist scan values so collect/plot don't depend on config staying
    # in sync with the inputs that were actually generated.
    np.save(f"{GRID_DIR}/d1_values.npy", d1_values)
    np.save(f"{GRID_DIR}/d2_values.npy", d2_values)

    # Copy the starting geometry into the working directory so ORCA can
    # find it (jobs run with cwd = GRID_DIR).
    os.system(f"cp {XYZ_FILE} {GRID_DIR}/")

    # --- Generate one .inp per grid point ---
    job_names = []
    for i, d1 in enumerate(d1_values):
        for j, d2 in enumerate(d2_values):
            generate_orca_input(d1, d2, i, j)
            job_names.append(f"point_{i:02d}_{j:02d}")
            print(f"Generated: point_{i:02d}_{j:02d}.inp  (d1={d1:.3f}, d2={d2:.3f})")

    # --- SLURM array job script ---
    # The array index maps into a flat list of job names.  SLURM sets
    # $SLURM_ARRAY_TASK_ID to the array element, which we use to look
    # up the corresponding input file.
    with open(f"{GRID_DIR}/submit_array.sh", 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH -p {PARTITION}
#SBATCH -A {ACCOUNT}
#SBATCH -J 2d_pes
#SBATCH -o slurm_%A_%a.out
#SBATCH -e slurm_%A_%a.err
#SBATCH -N 1
#SBATCH -n {NPROCS}
#SBATCH -t {WALLTIME}
#SBATCH --mem=30G
#SBATCH --array=0-{len(job_names)-1}

module load orca/v6.1.0

JOBS=({' '.join(job_names)})
JOB=${{JOBS[$SLURM_ARRAY_TASK_ID]}}

cd $SLURM_SUBMIT_DIR

{ORCA_PATH} $JOB.inp > $JOB.out
""")

    # --- Individual submission fallback ---
    with open(f"{GRID_DIR}/submit_all_individual.sh", 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Alternative: submit each point as a separate job\n\n")
        for name in job_names:
            f.write(f"""sbatch --wrap="{ORCA_PATH} {name}.inp > {name}.out" \\
  -p {PARTITION} -A {ACCOUNT} -J {name} -N 1 -n {NPROCS} \\
  --mem=30G -t {WALLTIME} -o {name}.slurm.out -e {name}.slurm.err
""")

    print(f"\nGenerated {len(job_names)} input files in {GRID_DIR}/")
    print(f"\nTo submit all at once:  cd {GRID_DIR} && sbatch submit_array.sh")
    print(f"Or individually:        cd {GRID_DIR} && bash submit_all_individual.sh")


# ================================================================
# COLLECT ENERGIES
# ================================================================

def extract_energy(output_file):
    """
    Extract the final single-point energy from an ORCA output file.

    ORCA prints "FINAL SINGLE POINT ENERGY" after each SCF converges.
    In a geometry optimization, this line appears multiple times (once
    per optimization step); we want the *last* occurrence, which
    corresponds to the converged geometry.

    Parameters
    ----------
    output_file : str
        Path to an ORCA ``.out`` file.

    Returns
    -------
    float or None
        The final energy in Hartrees, or None if the line was never
        found (indicating the job crashed or is still running).
    """
    energy = None
    with open(output_file, 'r') as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[-1])
    return energy


def collect_energies():
    """
    Parse all ORCA output files and assemble the 2D energy surface.

    Steps:
      1. Load the saved d₁/d₂ scan arrays.
      2. Loop over every grid point; extract the final energy from
         each ``.out`` file (if it exists and contains a valid energy).
      3. Convert absolute energies (Hartrees) to relative energies (eV)
         by subtracting the global minimum and multiplying by 27.211
         (the Hartree-to-eV conversion factor).
      4. Save the result as ``energies_eV.npy``.
      5. Clean up ORCA scratch files (gbw, tmp, engrad, etc.) to free
         disk space.  The ``.out`` files are kept for reference.

    Returns
    -------
    tuple of (ndarray, ndarray, ndarray)
        (d1_values, d2_values, energies_eV) — the scan arrays and the
        2D energy surface in eV.
    """

    d1_values = np.load(f"{GRID_DIR}/d1_values.npy")
    d2_values = np.load(f"{GRID_DIR}/d2_values.npy")
    n1, n2 = len(d1_values), len(d2_values)

    # Initialize with NaN so missing points are obvious in plots.
    energies = np.full((n1, n2), np.nan)
    failed = []

    for i in range(n1):
        for j in range(n2):
            outfile = f"{GRID_DIR}/point_{i:02d}_{j:02d}.out"
            if os.path.exists(outfile):
                E = extract_energy(outfile)
                if E is not None:
                    energies[i, j] = E
                else:
                    failed.append(f"point_{i:02d}_{j:02d}")
            else:
                failed.append(f"point_{i:02d}_{j:02d}")

    # Convert Hartrees → relative eV
    E_min = np.nanmin(energies)
    energies_eV = (energies - E_min) * 27.211  # 1 Hartree = 27.211 eV

    np.save(f"{GRID_DIR}/energies_eV.npy", energies_eV)

    n_complete = np.count_nonzero(~np.isnan(energies))
    print(f"Collected {n_complete}/{n1*n2} energies")
    if failed:
        print(f"Failed/missing: {failed}")

    # --- Clean up ORCA scratch files ---
    # These can be large and are not needed once energies are extracted.
    import glob
    for ext in ['*.inp', '*.opt', '*.bibtex', '*.densitiesinfo',
                '*.engrad', '*.property.txt', '*.gbw', '*.tmp',
                '*.xyz', '*.sh', '*.cpcm*', '*.densities']:
        for f in glob.glob(f"{GRID_DIR}/{ext}"):
            os.remove(f)
    for f in glob.glob(f"{GRID_DIR}/slurm_*"):
        os.remove(f)
    print("Cleaned up temporary files. Kept: .out, .npy, plots")
    print(f"\n  If you used this script or found it helpful, please cite:")
    print(f"  Authors, \"Paper Title\",")
    print(f"  Manuscript in preparation (2026).")
    print(f"")

    return d1_values, d2_values, energies_eV


# ================================================================
# PLOT THE 2D PES
# ================================================================

def plot_pes():
    """
    Generate a filled contour plot of the 2D potential energy surface.

    The plot uses 30 evenly spaced contour levels from 0 to either
    2.0 eV or the surface maximum (whichever is smaller), which keeps
    the color scale focused on the chemically relevant energy range.

    Two reference geometries are marked with stars:
      - White star: canonical Watson–Crick geometry (short d₁, short d₂).
      - Red star:   rare tautomeric form (long d₁, long d₂).

    The coordinates of these markers should be updated if the script is
    adapted to a different system (see "Adapting to Other Systems" in
    the module docstring).

    Saves the figure to ``{GRID_DIR}/2d_pes.png`` at 300 DPI.
    """

    d1_values = np.load(f"{GRID_DIR}/d1_values.npy")
    d2_values = np.load(f"{GRID_DIR}/d2_values.npy")
    energies = np.load(f"{GRID_DIR}/energies_eV.npy")

    # Create 2D coordinate grids for contour plotting.
    # indexing='ij' ensures rows map to d1 and columns to d2.
    D1, D2 = np.meshgrid(d1_values, d2_values, indexing='ij')

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Cap color scale at 2.0 eV to avoid washing out detail near minima
    levels = np.linspace(0, min(np.nanmax(energies), 2.0), 30)
    contour = ax.contourf(D1, D2, energies, levels=levels, cmap='viridis')
    ax.contour(D1, D2, energies, levels=levels, colors='black',
               linewidths=0.3, alpha=0.5)

    plt.colorbar(contour, ax=ax, label='Energy (eV)')

    ax.set_xlabel(r'$d_1$ — H5–N22 distance (Å)', fontsize=12)
    ax.set_ylabel(r'$d_2$ — H4–N21 distance (Å)', fontsize=12)
    ax.set_title('2D PES: Double Proton Transfer in G–C', fontsize=14)

    # Mark reference geometries.
    # UPDATE these coordinates if adapting to a different system.
    ax.plot(1.030, 1.034, 'w*', markersize=15, label='Canonical')
    ax.plot(1.870, 1.713, 'r*', markersize=15, label='Tautomeric')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{GRID_DIR}/2d_pes.png", dpi=300)
    plt.show()
    print(f"Saved plot to {GRID_DIR}/2d_pes.png")
    print(f"\n  If you used this script or found it helpful, please cite:")
    print(f"  Authors, \"Paper Title\",")
    print(f"  Manuscript in preparation (2026).")
    print(f"")


# ================================================================
# RESTART FAILED JOBS
# ================================================================

def find_failed_points():
    """
    Identify grid points that did not produce a valid energy.

    A point is considered "failed" if:
      - Its ``.out`` file does not exist (job never ran or was lost), OR
      - Its ``.out`` file exists but does not contain the string
        "FINAL SINGLE POINT ENERGY" (ORCA crashed before converging).

    Returns
    -------
    list of tuple
        Each element is (job_name, i, j) where job_name is the base
        filename (e.g. "point_03_17") and (i, j) are the grid indices.
    """
    d1_values = np.load(f"{GRID_DIR}/d1_values.npy")
    d2_values = np.load(f"{GRID_DIR}/d2_values.npy")

    failed = []
    for i in range(len(d1_values)):
        for j in range(len(d2_values)):
            name = f"point_{i:02d}_{j:02d}"
            outfile = f"{GRID_DIR}/{name}.out"
            ok = False
            if os.path.exists(outfile):
                with open(outfile, 'r') as f:
                    for line in f:
                        if "FINAL SINGLE POINT ENERGY" in line:
                            ok = True
                            break
            if not ok:
                failed.append((name, i, j))
    return failed


def restart_failed():
    """
    Locate failed or incomplete grid points and write resubmission scripts.

    This is useful after a batch run where some jobs timed out, crashed,
    or were killed by the scheduler.  The function:

      1. Scans all output files to find points without a valid energy.
      2. Regenerates ``.inp`` files for those points (in case the
         originals were cleaned up by ``collect``).
      3. Writes ``resubmit_array.sh`` and ``resubmit_individual.sh``
         targeting only the failed points.
    """
    d1_values = np.load(f"{GRID_DIR}/d1_values.npy")
    d2_values = np.load(f"{GRID_DIR}/d2_values.npy")

    failed = find_failed_points()

    if not failed:
        print("All points completed successfully! Nothing to restart.")
        return

    print(f"Found {len(failed)} failed/incomplete points:")
    for name, i, j in failed:
        print(f"  {name}  (d1={d1_values[i]:.3f}, d2={d2_values[j]:.3f})")

    # Regenerate input files for failed points if they were cleaned up
    for name, i, j in failed:
        inpfile = f"{GRID_DIR}/{name}.inp"
        if not os.path.exists(inpfile):
            generate_orca_input(d1_values[i], d2_values[j], i, j)
            print(f"  Regenerated {name}.inp")

    # Ensure the starting geometry is present
    if not os.path.exists(f"{GRID_DIR}/{XYZ_FILE}"):
        os.system(f"cp {XYZ_FILE} {GRID_DIR}/")

    job_names = [name for name, i, j in failed]

    # --- Array job for failed points only ---
    with open(f"{GRID_DIR}/resubmit_array.sh", 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH -p {PARTITION}
#SBATCH -A {ACCOUNT}
#SBATCH -J 2d_pes_restart
#SBATCH -o slurm_%A_%a.out
#SBATCH -e slurm_%A_%a.err
#SBATCH -N 1
#SBATCH -n {NPROCS}
#SBATCH -t {WALLTIME}
#SBATCH --mem=30G
#SBATCH --array=0-{len(job_names)-1}

module load orca/v6.1.0

JOBS=({' '.join(job_names)})
JOB=${{JOBS[$SLURM_ARRAY_TASK_ID]}}

cd $SLURM_SUBMIT_DIR

{ORCA_PATH} $JOB.inp > $JOB.out
""")

    # --- Individual submission fallback ---
    with open(f"{GRID_DIR}/resubmit_individual.sh", 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Resubmit {len(job_names)} failed points\n\n")
        for name in job_names:
            f.write(f"""sbatch --wrap="{ORCA_PATH} {name}.inp > {name}.out" \\
  -p {PARTITION} -A {ACCOUNT} -J {name} -N 1 -n {NPROCS} \\
  --mem=30G -t {WALLTIME} -o {name}.slurm.out -e {name}.slurm.err
""")

    print(f"\nGenerated resubmit scripts for {len(job_names)} points.")
    print(f"  Array job:    cd {GRID_DIR} && sbatch resubmit_array.sh")
    print(f"  Individual:   cd {GRID_DIR} && bash resubmit_individual.sh")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generate_2d_pes.py generate   # Create inputs + SLURM scripts")
        print("  python generate_2d_pes.py collect     # Extract energies from outputs")
        print("  python generate_2d_pes.py plot        # Contour plot of the 2D PES")
        print("  python generate_2d_pes.py restart     # Resubmit failed points")
        sys.exit(1)

    action = sys.argv[1]

    if action == "generate":
        generate_all_inputs()
    elif action == "collect":
        collect_energies()
    elif action == "plot":
        plot_pes()
    elif action == "restart":
        restart_failed()
    else:
        print(f"Unknown action: {action}")
