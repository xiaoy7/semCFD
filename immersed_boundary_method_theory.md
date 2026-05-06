# Immersed Boundary Method Theory in semCFD

This note summarizes the immersed boundary method (IBM) theory used by the
current 2D semCFD implementation, especially the Baby Spider Shower case.
The implementation follows the main ideas of Peskin's immersed boundary
method and the IB2d `IBM_Driver.m` time-stepping structure, while keeping
the fluid solve in semCFD's spectral element projection framework.

## 1. Physical Model

The fluid is modeled by the incompressible Navier-Stokes equations with an
immersed-boundary force:

```text
du/dt + u . grad(u) = -grad(p) + nu Laplacian(u) + f_ib + f_ext
div(u) = 0
```

where:

- `u = (u, v)` is the Eulerian fluid velocity.
- `p` is the pressure or pressure correction variable.
- `nu` is the kinematic viscosity.
- `f_ib` is the force spread from the Lagrangian immersed structure.
- `f_ext` is any additional body force, such as the optional flow-driving
  force used to mimic IB2d examples.

The immersed structure is represented by Lagrangian marker points:

```text
X_l(t) = (x_l(t), y_l(t)),   l = 1, ..., Nb
```

These points do not live on the Eulerian SEM grid. They move through the
fluid and exchange force and velocity with the Eulerian grid through a
regularized delta function.

## 2. Lagrangian-Eulerian Coupling

IBM has two fundamental coupling operations.

### 2.1 Force Spreading

The Lagrangian structure computes force density `F_lag_l` at its marker
points. This force is spread to the Eulerian grid:

```text
f_ib(x_q) = sum_l F_lag_l delta_h(x_q - X_l) ds_l
```

where:

- `x_q` is an Eulerian SEM node.
- `X_l` is the Lagrangian marker position.
- `ds_l` is the Lagrangian marker spacing.
- `delta_h` is a regularized delta kernel.

In the code this is handled by:

```text
ibm_lagrangian_forces2d.m
ibm_spread_lagrangian_force2d.m
```

### 2.2 Velocity Interpolation

The Lagrangian markers move with the local fluid velocity:

```text
dX_l/dt = U_lag_l
U_lag_l = integral u(x) delta_h(x - X_l) dx
```

In discrete form:

```text
U_lag_l ~= sum_q u(x_q) delta_h(x_q - X_l) dA_q
```

The current semCFD implementation uses interpolation from the SEM tensor
grid to marker locations through `griddedInterpolant`, inside:

```text
ibm_begin_lagrangian_step2d.m
ibm_update_lagrangian_markers2d.m
```

## 3. Regularized Delta Function

The implementation uses Peskin's 4-point delta kernel in 1D:

```text
phi(r) =
    (3 - 2r + sqrt(1 + 4r - 4r^2)) / 8,      0 <= r < 1
    (5 - 2r - sqrt(-7 + 12r - 4r^2)) / 8,    1 <= r < 2
    0,                                       r >= 2
```

The dimensional 1D kernel is:

```text
delta_h(x) = phi(|x| / h) / h
```

The 2D tensor-product kernel is:

```text
delta_h(x, y) = delta_hx(x) delta_hy(y)
```

In semCFD this is implemented in:

```text
ibm_delta_peskin4_1d.m
```

The spreading operation normalizes the kernel using the local SEM nodal
control-volume weights to reduce loss of force near nonuniform nodal
spacing.

## 4. Structural Force Models

The Lagrangian structure can be built from the same basic force models used
in IB2d.

### 4.1 Springs

A spring connects marker `i` to marker `j`. Let:

```text
dX = X_j - X_i
L = |dX|
L0 = resting length
k = spring stiffness
alpha = spring nonlinearity exponent
```

The spring force magnitude is:

```text
F = 0.5 (alpha + 1) k (L - L0)^alpha
```

The vector force applied to marker `i` is:

```text
F_i = F dX / L
```

and marker `j` receives the equal and opposite force.

### 4.2 Beams

The torsional beam model uses three markers `(p, q, r)` and penalizes a
discrete curvature-like quantity:

```text
C_current = (x_r - x_q)(y_q - y_p) - (y_r - y_q)(x_q - x_p)
```

The beam force is proportional to:

```text
k_beam (C_current - C_reference)
```

This is the same type of torsional-beam model used by IB2d.

### 4.3 Target Points

Target points tether a Lagrangian marker to a prescribed position:

```text
F_i = k_target (X_target_i - X_i)
```

They are useful for fixed or prescribed immersed boundaries. In the Baby
Spider Shower case, targets are disabled because the web is intended to move.

### 4.4 Muscle Model

The code also contains a simplified length-tension / force-velocity muscle
model. For a muscle connection `(i, j)`:

```text
L = |X_j - X_i|
L_opt = optimal muscle length
activation = a(t)
F_max = maximum muscle force
```

The force is modeled as:

```text
F_muscle = activation * F_max * length_tension(L) * force_velocity(dL/dt)
```

This is present in the code but disabled for the Baby Spider Shower setup.

## 5. IBM Time Stepping

The semCFD IBM update follows the midpoint-style structure of IB2d's
`IBM_Driver.m`.

At time step `n`:

### Step 1: Predict Lagrangian Half Step

```text
X_h = X_n + 0.5 dt U_n(X_n)
```

Implemented in:

```text
ibm_begin_lagrangian_step2d.m
```

### Step 2: Compute Lagrangian Force at Half Step

```text
F_lag_h = F_structure(X_h)
```

Implemented in:

```text
ibm_lagrangian_forces2d.m
```

### Step 3: Spread Force to Eulerian Grid

```text
f_ib_h(x_q) = sum_l F_lag_h,l delta_h(x_q - X_h,l) ds_l
```

Implemented in:

```text
ibm_spread_lagrangian_force2d.m
```

### Step 4: Advance the Fluid

The IBM force is added to the explicit RHS:

```text
U_star = u_n / dt - convection + f_ib
```

The pressure projection and implicit Helmholtz solve are then handled by
semCFD's SEM tensor-product solver.

### Step 5: Move Lagrangian Points

After computing the new fluid velocity:

```text
X_{n+1} = X_n + dt U_{n+1}(X_h)
```

Implemented in:

```text
ibm_update_lagrangian_markers2d.m
```

## 6. Skew-Symmetric Convection

The fluid solver uses a skew-symmetric convection form:

```text
N_u = 0.5 * (u u_x + v u_y + (u^2)_x + (uv)_y)
N_v = 0.5 * (u v_x + v v_y + (uv)_x + (v^2)_y)
```

This is important for stability. The plain advective form:

```text
u u_x + v u_y
```

can inject artificial kinetic energy in long runs. The skew form is closer
to IB2d's fluid update and prevents late-time numerical divergence.

## 7. Baby Spider Shower Case

The Baby Spider Shower case is configured in:

```text
ibm_setup_baby_spider_shower2d.m
```

It represents a vertical spider web with:

- 90 Lagrangian points.
- 89 springs.
- 88 torsional beams.
- One massive endpoint.
- Gravity acting downward.
- Smooth through-flow boundary velocity.

The Lagrangian spacing follows the IB2d example:

```text
ds = Lx / (2 * Nx)
```

For the current semCFD grid:

```text
Lx = 1.5
Nx = para.nx_all - 1 = 384
ds = 1.5 / (2 * 384) = 0.001953125
```

This matches the IB2d `spider.vertex`, `spider.spring`, and `spider.beam`
files.

The web is not target-fixed. Therefore, its Lagrangian markers are expected
to move with the fluid. If a fixed web is desired, target points must be
enabled or the marker update must be disabled for the desired marker subset.

## 8. VTK Output

The semCFD IBM output writes IB2d-style VTK files into:

```text
viz_IB2d/
```

Typical files are:

```text
u.0001.vtk              Eulerian velocity vector
P.0001.vtk              pressure
Omega.0001.vtk          vorticity
uMag.0001.vtk           velocity magnitude
fX.0001.vtk             IBM/body-force x component
fY.0001.vtk             IBM/body-force y component
fMag.0001.vtk           force magnitude
lagsPts.0001.vtk        Lagrangian marker points
lagPtsConnect.0001.vtk  Lagrangian marker points with spring connectivity
```

The VTK output is implemented in:

```text
sem_write_ibm_vtk2d.m
```

## 9. Main Differences from IB2d

The semCFD implementation keeps the IBM theory but changes the fluid
discretization:

- IB2d uses a uniform finite-difference grid and FFT-based periodic fluid
  solver.
- semCFD uses spectral element derivative matrices and a projection /
  Helmholtz solve.
- IB2d's Baby Spider example drives flow with an internal target-velocity
  forcing strip in a periodic box.
- semCFD instead uses a smooth boundary-driven through-flow to avoid trapping
  momentum in a closed no-slip domain.

The essential IBM coupling remains the same:

```text
Lagrangian force -> regularized delta spreading -> Eulerian fluid solve
Eulerian velocity -> interpolation -> Lagrangian marker motion
```

