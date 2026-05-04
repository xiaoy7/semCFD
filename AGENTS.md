# semCFD Agent Notes

## Project Overview

`semCFD` is a MATLAB research codebase for CFD solvers based on tensor-product
spectral element methods (SEM). The repository includes Poisson/Helmholtz
building blocks, projection-method Navier-Stokes solvers, Cahn-Hilliard phase
field solvers, coupled NS-CH two-phase flow scripts, and recent 2D immersed
boundary method (IBM) fluid-structure experiments.

The project is script-driven. There is no package manager file or unified test
suite. Many main scripts copy all `*.m` files into a sibling
`timeYYYYMMDD_HHmm_ss` directory at startup so that each run keeps a source
snapshot together with output files.

## Main Entry Points

- `xy_ns_ibm_2d.m`: Most active 2D Navier-Stokes + IBM driver. The default case
  is `baby_spider_shower`. Use `SEMCFD_IBM_CASE` to choose a case and
  `SEMCFD_STEPS` to override the step count. It uses 2D tensor-product SEM,
  pressure projection, CPU/GPU branches, Tecplot output, and VTK IBM output.
- `xy_nsch_2d.m`: 2D Navier-Stokes/Cahn-Hilliard two-phase flow solver. See
  `xy_nsch_2d_algorithm.md` for the algorithm notes.
- `xy_ch_2d.m`: 2D Cahn-Hilliard phase-field solver.
- `xy_ns_3d.m`: 3D lid-driven cavity Navier-Stokes projection solver.
- `xy_nsch_3d.m`: 3D NS-CH two-phase flow script.
- `two_phase_sem_tensor_demo.m`: Compact 2D NS-CH tensor-product demo, useful
  as a readable prototype.
- `ch_mms_convergence_2d.m`: Manufactured-solution convergence test for the 2D
  Cahn-Hilliard solver.

## Numerical Methods And Data Layout

- Spatial discretization is mainly continuous Galerkin SEM on
  Legendre-Gauss-Lobatto nodes.
- 2D fields are usually stored as `(x_index, y_index)` arrays. A common grid
  construction is `[coordY, coordX] = meshgrid(y, x)`.
- 3D fields are usually stored as `(x_index, y_index, z_index)` arrays and use
  `tensorprod` and `pagemtimes` heavily.
- Dirichlet velocity unknowns are often solved only on interior/free nodes.
  Pressure and phase correction variables often use full-node Neumann grids.
- Constant-coefficient Poisson/Helmholtz problems are solved by 1D generalized
  eigendecompositions of the tensor-product Laplacian. In modal space, each
  coefficient is divided by `alpha + lambda_x + lambda_y (+ lambda_z)`.
- Pure Neumann pressure solves usually regularize the constant mode by setting
  the `(1,1)` denominator to `1`, then subtracting the mean afterward.

## Core Utility Files

- `parameter_bc2d.m`: Sets 2D Dirichlet/Neumann boundary nodes and free nodes.
- `parameter_bc.m`: 3D boundary-node version.
- `cal_matrix2.m`: Builds 1D SEM grid, differentiation matrix, modal transform,
  and eigenvalues for Dirichlet/Neumann/periodic cases.
- `cal_matrix2_1.m`: Lightweight matrix builder used for Neumann/pressure grids.
- `LegendreD.m`, `LegendreGL.m`, `Jacobi*.m`: GLL/Jacobi nodes, weights, and
  derivative helpers.
- `calDerivativeMatrix.m`: Assembles the global first-derivative matrix.
- `assemble_sem_weights.m`: Builds 1D SEM quadrature weights, often used for
  mass weighting and pressure boundary correction.
- `calLaplace.m`, `calLaplace3D.m`, `applyLap3D.m`: Laplacian application
  helpers.
- `sem_tensor_ops.m`: Static class with 3D gradient, divergence, and Laplacian
  tensor operations.

## IBM Module

Many IBM-related files are recent and some are currently untracked. Do not
delete or roll them back unless the user explicitly asks.

- `ibm_setup2d.m`: Builds the default marker-based IBM model. Supports
  `lagrangian_structure` and `marker_direct_forcing`.
- `ibm_setup_baby_spider_shower2d.m`: Case-specific model setup for
  `baby_spider_shower`.
- `ibm_begin_lagrangian_step2d.m`, `ibm_update_lagrangian_markers2d.m`:
  Lagrangian marker time stepping.
- `ibm_lagrangian_forces2d.m`: Springs, beams, targets, masses, and muscle
  forces on Lagrangian markers.
- `ibm_spread_lagrangian_force2d.m`: Spreads Lagrangian forces to the Eulerian
  grid.
- `ibm_direct_forcing2d.m`: Entry point for rigid direct-forcing or Lagrangian
  structure forcing.
- `ibm_update_rigid_body2d.m`, `ibm_keep_body_inside_domain2d.m`: Rigid-body
  motion and domain constraints.
- `ibm_refresh_markers2d.m`, `ibm_refresh_mask2d.m`: Marker and Eulerian mask
  refresh helpers.
- `ibm_delta_peskin4_1d.m`: Peskin 4-point delta kernel.
- `sem_write_ibm_vtk2d.m`: Writes IB2d-style VTK files for velocity, pressure,
  vorticity, force fields, and Lagrangian points.

## Output Files

- `OUTPUT_Tecplot2D4.m`, `OUTPUT_Tecplot2D5.m`: 2D Tecplot ASCII output.
- `OUTPUT_Tecplot3d.m`, `OUTPUT_Tecplot3d2.m`, `exportTecplot3D.m`: 3D Tecplot
  output.
- `plt_Head*.m`, `plt_Zone.m`: Tecplot header/zone helpers.
- `sem_write_ibm_vtk2d.m`: IBM VTK output, usually written to `viz_IB2d` under
  the run archive directory.
- Main scripts often create `time...` directories and copy source files there.
  Treat these as run artifacts, not algorithm changes.

## Running

Typical MATLAB usage:

```matlab
cd('D:\3d\current\semCFD\semCFD')
xy_ns_ibm_2d
```

Several scripts contain this hard-coded dependency path:

```matlab
addpath(genpath("D:\semMatlab"))
```

On another machine, confirm that path exists or adjust it to a local helper
path. The code usually checks `gpuDeviceCount('available')`; if a GPU is
available, many arrays are moved to `gpuArray`, otherwise the CPU path is used.

For a short IBM run from PowerShell:

```powershell
$env:SEMCFD_STEPS='100'
matlab -batch "cd('D:\3d\current\semCFD\semCFD'); xy_ns_ibm_2d"
```

## Verification

- Use MATLAB `checkcode` after editing `.m` files to catch syntax and style
  issues quickly.
- `ch_mms_convergence_2d` is the clearest numerical verification entry point,
  but its default sweeps are large. For debugging, reduce the grid sizes or
  time steps first.
- IBM/NS scripts can be long-running. Prefer `SEMCFD_STEPS` for smoke tests.
- When editing GPU branches, keep CPU behavior in sync and check `gather`,
  `gpuArray`, and `wait(Device)` usage.

## Current Worktree Notes

At the time this file was created, the worktree already had modified and
untracked files, mostly around the IBM work. These appear to be active
experimental changes:

- Modified: `ibm_direct_forcing2d.m`, `ibm_setup2d.m`,
  `ibm_update_rigid_body2d.m`, `xy_ns_ibm_2d.m`
- Untracked: many `ibm_*2d.m` files, `sem_write_ibm_vtk2d.m`,
  `xy_ns_ibm_2d1.m`, `xy_ns_ibm_2d2.m`, and several `.bak_codex` backups

Future agents should avoid `git reset --hard`, `git checkout -- <file>`, or
deleting backup files unless the user explicitly requests it.

## Editing Conventions

- Preserve the existing MATLAB script style and naming conventions. Avoid broad
  refactors unless they are necessary for the requested change.
- Be careful around SEM matrix assembly, boundary-node logic, pressure
  projection, and IBM force spread/interpolation. These changes affect solver
  stability globally.
- When adding output variables, update Tecplot/VTK variable lists, dimensions,
  and `gather` logic together.
- When changing GPU logic, check the CPU path as well.
- Avoid adding uncontrolled randomness to driver scripts. If randomness is
  needed, set and record an explicit seed.
- Running main scripts can generate large `time...` output directories. Do not
  add large run artifacts to git.

