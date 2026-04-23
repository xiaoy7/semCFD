# Coupled NS–Cahn–Hilliard Solver (`xy_nsch_2d.m`)

This note summarizes the governing theory and numerical algorithm implemented in `xy_nsch_2d.m`. The code advances an incompressible two-phase flow with surface tension using a projection-based Navier–Stokes (NS) solver coupled to a Cahn–Hilliard (CH) phase-field model, both posed on a tensor-product spectral element mesh.

## Governing Equations

### Phase Field
- Order parameter `φ(x,t)` distinguishes the two fluids: `φ ≈ +1` in the light phase and `φ ≈ -1` in the heavy phase.
- The Cahn–Hilliard system reads  
  \[
  \partial_t \phi + \mathbf{u}\cdot\nabla\phi = \gamma_1 \nabla^2 \psi,
  \qquad
  \psi = \frac{1}{\eta^2}\left(\phi^3 - \phi\right) - \nabla^2\phi,
  \]
  where \(\psi\) is the chemical potential, \(\gamma_1\) is the mobility, and \(\eta\) controls interface thickness.
- Surface tension enters the momentum equation through the Korteweg force  
  \[
  \mathbf{f}_{\sigma} = \sigma \lambda\, \psi \nabla \phi,
  \]
  with \(\sigma\) the physical surface tension and  \(\lambda = 3\sqrt{2}\eta  / 4 \) the phase-field coefficient used in the code.

### Mixture Properties
- Density and dynamic viscosity vary smoothly with \(\phi\):  
  \[
  \rho(\phi) = \frac{\rho_H + \rho_L}{2} + \frac{\rho_L - \rho_H}{2}\phi,\qquad
  \mu(\phi) = \frac{\mu_H + \mu_L}{2} + \frac{\mu_L - \mu_H}{2}\phi,
  \]
  where subscripts `H` and `L` denote the heavy and light fluids.
- Auxiliary gradients of \(\mu\) are retained to capture viscous stress variations inside the diffuse interface.

### Fluid Momentum
- The incompressible Navier–Stokes equations are written in projection form:  
  \[
  \rho \left(\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}\right)
  = -\nabla p + \nabla\cdot\left[\mu\left(\nabla\mathbf{u} + \nabla\mathbf{u}^\top\right)\right] + \mathbf{f}_{\sigma} + \rho \mathbf{g}.
  \]
- A constant reference density \(\rho_0\) is introduced in the pressure projection to keep the Poisson problem well-conditioned.
- Mass conservation is enforced by solving a pressure correction step ensuring \(nabla·u^{n+1} = 0\)

## Spatial Discretization
- Tensor-product spectral elements (Legendre–Gauss–Lobatto points) are used in both `x` and `y`.
- `cal_matrix2` builds the SEM differentiation matrices (`Dmatrixx`, `Dmatrixy`) and eigenvalues (\(\lambda_{xd}, \lambda_{yd}\)) for Dirichlet velocity unknowns.
- `cal_matrix2_1` builds the analogous matrices on a Neumann grid for pressure/phase unknowns.
- Helmholtz/Poisson solves are diagonalized in spectral space using modal transforms (`T`, \(T^{-1}\)) and `tensorprod`.
- Boundary conditions:
  - Velocity: homogeneous Dirichlet (no-slip) except the lid velocity that can be prescribed on the top boundary (`lid_velocity`).
  - Pressure: homogeneous Neumann, with the mean removed to fix the null space.

## Tensor-Product Laplacian

Because each basis function is a tensor product of one-dimensional SEM shape functions, the discrete Laplacian separates as a Kronecker sum. Denote the one-dimensional mass and stiffness matrices by \(M_x, S_x\) in the horizontal direction and \(M_y, S_y\) vertically. For nodal coefficient arrays \(U\) and \(V\) (with `vec` stacking grid values), the bilinear form can be written as
\[
\mathcal{A}_h(U,V) = (S_x \otimes M_y + M_x \otimes S_y) \, \mathrm{vec}(U) \cdot \mathrm{vec}(V).
\]
The one-dimensional generalized eigenvalue problems constructed in `cal_matrix2` provide modal matrices \(T_x, T_y\) such that
\[
H_x = M_x^{-1} S_x = T_x \Lambda_x T_x^{-1}, \qquad
H_y = M_y^{-1} S_y = T_y \Lambda_y T_y^{-1},
\]
with diagonal matrices \(\Lambda_x = \operatorname{diag}(\lambda_x)\) and \(\Lambda_y = \operatorname{diag}(\lambda_y)\). Consequently, applying the Laplacian to the coefficient matrix \(U\) becomes
\[
\nabla_h^2 U = H_x U + U H_y^\top,
\]
which the code evaluates through `calLaplace` as `Dmatrixx2 * U + U * DmatrixyT2`.

For Helmholtz or Poisson solves, transforming into modal space decouples the operator:
\[
\widehat{U} = T_x^{-1} U T_y^{-T}, \qquad
(\alpha - \nabla_h^2)\widehat{U} = \left(\alpha + \Lambda_x \mathbf{1}^\top + \mathbf{1} \Lambda_y^\top\right)\widehat{U}.
\]
Each spectral coefficient is updated via a simple division by the preassembled tensor \(\alpha + \lambda_x + \lambda_y\), and transforming back with \(T_x\) and \(T_y\) yields the physical-space solution. This tensor-product structure is key to the efficiency of all elliptic solves (`helmholtz_u`, `helmholtz_v`, `helmholtzPhi`, `helmholtzPsi`) in `xy_nsch_2d.m`.

## Temporal Integration

All fields are advanced with a two-step scheme combining Adams–Bashforth extrapolation for convective terms and BDF-like centering for diffusive terms. The discrete operators \(D_x\), \(D_y\), \(∇·\), and \(∇²\) correspond to the SEM differentiation matrices assembled in the code.

### Step-by-step Equations

1. **Temporal extrapolation**  
   Build first- and second-order extrapolations of the solution:
   \[
   \begin{aligned}
   \mathbf{u}^\ast &= 2\mathbf{u}^n - \mathbf{u}^{n-1}, \qquad
   \phi^\ast = 2\phi^n - \phi^{n-1}, \\
   \widehat{\mathbf{u}} &= 2\mathbf{u}^n - \tfrac{1}{2}\mathbf{u}^{n-1}, \qquad
   \widehat{\phi} = 2\phi^n - \tfrac{1}{2}\phi^{n-1}, \\
   p^\ast &= 2p^n - p^{n-1}.
   \end{aligned}
   \]
   These are implemented in `uStar`, `phiStar`, `uCap`, and `pStar`.

2. **Phase-field advection-diffusion**  
   Compute the chemical potential forcing
   \[
   Q_1 = \frac{\mathbf{u}^\ast\cdot\nabla\phi^\ast - \widehat{\phi}/\Delta t}{\gamma_1},
   \qquad
   Q_2 = \frac{\left(\phi^\ast\right)^2 - 1 - S}{\eta^2}\,\phi^\ast,
   \]
   where `S = bigs` is the stabilization parameter.  
   Solve the Helmholtz problems
   \[
   \left(\alpha + \frac{S}{\eta^2} - \nabla^2\right)\psi^{n+1/2} = Q_1 - \nabla^2 Q_2,
   \]
   \[
   \left(\alpha - \nabla^2\right)\phi^{n+1} = \psi^{n+1/2},
   \]
   by transforming to spectral space. Afterwards clamp \(\phi^{n+1}\) to `[-1,1]`.

3. **Mixture properties and capillary force**  
   Reconstruct the material properties
   \[
   \rho^{n+1} = \frac{\rho_H + \rho_L}{2} + \frac{\rho_L - \rho_H}{2}\phi^{n+1},\qquad
   \mu^{n+1} = \frac{\mu_H + \mu_L}{2} + \frac{\mu_L - \mu_H}{2}\phi^{n+1},
   \]
   and their gradients through \(D_x \phi^{n+1}\), \(D_y \phi^{n+1}\).  
   The Korteweg force is assembled as
   \[
   \mathbf{f}_\sigma^{n+1} = \sigma \lambda\,\psi_1^{n+1}\nabla\phi^{n+1}, \quad
   \psi_1^{n+1} = \frac{1}{\eta^2}\left((\phi^{n+1})^3 - \phi^{n+1}\right) - \nabla^2\phi^{n+1}.
   \]

4. **Momentum predictor**  
   Form the viscous and capillary contributions in each direction:
   \[
   \begin{aligned}
   &D\mu_x = \tfrac{1}{2}(\mu_H - \mu_L) D_x \phi^{n+1},\qquad
   D\mu_y = \tfrac{1}{2}(\mu_H - \mu_L) D_y \phi^{n+1},\\
   &\mathcal{L}_u = \nabla^2 \mathbf{u}^\ast, \qquad
   \mathcal{N}_u = (\mathbf{u}^\ast\cdot\nabla)\mathbf{u}^\ast.
   \end{aligned}
   \]
   The discrete right-hand side for the intermediate momentum is
   \[
   \mathbf{g}^{n+1} =
   \frac{\widehat{\mathbf{u}}}{\Delta t}  - \mathcal{N}_u  + (\nu - \nu_0)\mathcal{L}_u   + \frac{1}{\rho^{n+1}}\mathbf{f}_\sigma^{n+1}  + \frac{1}{\rho^{n+1}}\mathbf{f}_{\mu}^{n+1}   + \mathbf{g},
   \]
   where \(nu = \mu / \rho\) and `ν₀ = nium`. The extra viscous term
   \[
   \mathbf{f}_{\mu}^{n+1} =
   \begin{bmatrix}
   2 D\mu_x D_x u^\ast + D\mu_y (D_x v^\ast + D_y u^\ast) \\
   D\mu_x (D_x v^\ast + D_y u^\ast) + 2 D\mu_y D_y v^\ast
   \end{bmatrix}
   \]
   captures variable-viscosity stresses.  
   Pressure-gradient contributions from \(p^\ast\) are included through
   \[
   \mathbf{g}^{n+1} \leftarrow \mathbf{g}^{n+1} + \left(\frac{1}{\rho_0} - \frac{1}{\rho^{n+1}}\right)\nabla p^\ast,
   \]
   yielding the predictor fields `uv31x` and `uv31y`.  
   The intermediate velocity \(\tilde{\mathbf{u}}\) is then obtained by solving
   \[
   \left(\frac{\gamma_0}{\Delta t} - \nu_0 \nabla^2\right)\tilde{\mathbf{u}} = \mathbf{g}^{n+1},
   \]
   separately for each velocity component using spectral Helmholtz inversion with `γ₀ = 1.5` and `ν₀ = nium`.

5. **Pressure Poisson problem**  
   Enforce incompressibility on the predicted momentum by solving for the pressure correction `\varphi` (distinct from the phase field):
   \[
   -\rho_0\,\nabla\cdot \mathbf{g}^{n+1} = \rho_0\,\nabla^2 \varphi,
   \]
   supplemented with Neumann boundary conditions that account for momentum fluxes across the domain boundaries. The discrete rhs is assembled as
   \[
   f_{\text{solver}} = -\rho_0\,\nabla\cdot \mathbf{g}^{n+1} + \frac{F_g}{M},
   \]
   where `F_g` is the boundary correction and `M = wx ⊗ wy` is the diagonal mass matrix. Spectral diagonalization gives
   \[
   \nabla^2 \varphi = \frac{\gamma_0 \rho_0}{\Delta t}\nabla\cdot \tilde{\mathbf{u}},
   \]
   and subtracting the mean fixes the null space.

6. **Velocity correction (projection)**  
   Recover the pressure increment and correct the velocity:
   \[
   \varphi = \text{Poisson}(f_{\text{solver}}),\qquad
   p^{n+1} = p^\ast + \varphi,
   \]
   \[
   \mathbf{u}^{n+1} = \tilde{\mathbf{u}} - \frac{\Delta t}{\gamma_0}\frac{1}{\rho^{n+1}}\nabla \varphi.
   \]
   Here `γ₀ = 1.5` matches the BDF coefficient used in the code, and division by the local density retains correct capillary pressure jumps.

7. **Diagnostics and output**  
   Evaluate the residual
   \[
   \mathcal{E}^{n+1} = \lVert \mathbf{u}^{n+1} - \mathbf{u}^n \rVert_F,
   \]
   and export flow snapshots every `freOut` iterations.

## Notes on Implementation Details

- **GPU support**: If GPUs are available, matrices and fields are moved to device memory via `gpuArray`. CPU/GPU synchronization is handled with `wait(Device)`.
- **Mass matrix handling**: `mass_diag = wx * wy'` stores the tensor-product quadrature weights to apply boundary flux corrections in the pressure solve.
- **Copy-on-start**: The script archives all `.m` files in a timestamped directory (`timeYYYYMMDD_HHmm_ss`) before running, aiding reproducibility.
- **Stability considerations**:
  - `delta` (time step) must satisfy both CFL and diffusive limits due to advection, viscosity, and Cahn–Hilliard dynamics.
  - `aa` is checked to ensure the auxiliary parameter `α` remains real.

## Model Features & Innovations

- Tailored **spectral-element / Cahn–Hilliard coupling**: the code advances the phase field on Neumann nodes while the velocity uses Dirichlet nodes, linked through modal transforms so both solvers share a consistent spectral basis yet respect distinct boundary conditions.
- **Density-aware projection**: the velocity correction divides the pressure gradient by the local mixture density, maintaining the correct Laplace pressure jump without sacrificing the conditioning of the pressure Poisson problem (which still uses a constant reference density `ρ₀`).
- Incorporates **variable-viscosity stresses** by explicitly discretizing the `∇μ · (∇u + ∇uᵗ)` contributions, which are often neglected in standard diffuse-interface solvers, improving accuracy for large viscosity contrasts.
- Introduces a **stabilized Cahn–Hilliard update** using the `bigs` parameter, enabling larger time steps while retaining sharp interfaces.
- Provides a **modular GPU/CPU execution path**: the same script switches between `gpuArray` and double-precision arrays, facilitating rapid prototyping and accelerated production runs.
- Generates **Tecplot-ready outputs** that bundle velocity, pressure, composition, density, viscosity, and capillary forces, aiding validation and publication-quality visualization.

## References
- G. Tryggvason et al., _Direct Numerical Simulations of Two-Phase Flow_, Cambridge (2011) — diffuse-interface overview.
- M. D. Grzetic, _Spectral Element Methods for Incompressible Flow_ — reference for SEM discretization and projection.
- Dong & Shen (2012), _A Semi-Analytical Method for Two-Phase Flow_ — similar projection coupling with Cahn–Hilliard.

This markdown should serve as a quick reference when modifying `xy_nsch_2d.m` or extending it to new physics, geometries, or boundary conditions.
