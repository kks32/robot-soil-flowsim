# SWE JAX Simulator: Implemented Equations and Assumptions

This document describes the exact equations and numerical choices used in
`swe_simulator_jax.py`, including approximations and non-physical stabilizers.
All symbols are ASCII; units are SI unless noted.

## State and Parameters

- h(x,y): height (depth) of granular layer
- u(x,y), v(x,y): depth-averaged velocities
- hu = h*u, hv = h*v (momentum-like variables)

Key parameters:
- g: gravity
- rho_bulk: bulk density (constant, no dilation)
- rho_grain: grain density (for mu(I))
- grain_d: mean grain diameter
- mu_s, mu_2, I0: mu(I) parameters
- h_min: dry threshold
- tool_drag, tool_splash: DRFT coefficients
- tool_width, tool_length: rectangular tool footprint

## Governing Equations (Implemented)

### Mass Conservation

We advect height with a Rusanov flux:

```
dh/dt + d(hu)/dx + d(hv)/dy = 0
```

Fluxes use local Lax-Friedrichs (Rusanov) with speed = |u| or |v| in each
direction. This is intentionally diffusive but stable and differentiable.

### Momentum (Depth-Averaged)

We update velocities explicitly:

```
du/dt = -(u du/dx + v du/dy) - g * dh/dx * gate + a_fric_x + a_tool_x
dv/dt = -(u dv/dx + v dv/dy) - g * dh/dy * gate + a_fric_y + a_tool_y
```

`gate` is a smooth slope-trigger for avalanching:

```
slope = sqrt((dh/dx)^2 + (dh/dy)^2)
gate  = sigmoid((slope - mu_s) / slope_eps)
```

This is a heuristic yield-like behavior: slopes below mu_s do not drive flow.

### mu(I) Friction

We compute strain rate magnitude:

```
strain = sqrt(du/dx^2 + dv/dy^2 + 0.5*(du/dy + dv/dx)^2 + eps)
```

Pressure and inertial number:

```
P = rho_bulk * g * h
I = grain_d * strain / sqrt(P / rho_grain + eps)
```

Friction coefficient:

```
mu(I) = mu_s + (mu_2 - mu_s) / (I0 / (I + eps) + 1)
```

Friction acceleration:

```
a_fric = -g * mu(I) * v_hat
```

If `dt * |a_fric| > |v|`, we clamp to stop in one step (numerical stability).

### DRFT Tool Forcing (Momentum Source)

Tool is a soft rectangular footprint aligned with tool velocity. Let the tool
pose be (tx, ty, tz) and velocity (tvx, tvy).

Footprint mask (smooth sigmoid edges):

```
mask = sigmoid((L/2 - |long|)/s) * sigmoid((W/2 - |lat|)/s)
```

where `long` and `lat` are coordinates in the tool frame, `s` is softness.

Immersion:

```
z_imm = clip(h - tz, 0, h)
```

DRFT traction magnitude:

```
tau = z_imm * (C_drag + C_splash * rho_bulk * |v_tool|^2)
```

Acceleration source:

```
a_tool = tau / (rho_bulk * h) * v_tool_hat * mask
```

Tool motion is prescribed (kinematic). There is no tool dynamics or force
feedback loop in the simulator.

## Numerical Notes and Stabilizers

- Gradients: centered differences with edge padding.
- Time integration: explicit Euler.
- Dry handling: if h <= h_min, set velocity to zero.
- Speed cap: if |v| > speed_cap, rescale to cap (non-physical, stabilizer).
- Mass advection uses Rusanov flux; momentum uses advective form.

## Approximations and Limitations

1. Depth-averaged SWE (no vertical velocity or stress profile).
2. Constant bulk density (no dilation or compaction).
3. mu(I) used as a scalar friction coefficient; not a full viscoplastic stress.
4. Yielding is approximated by the slope gate; not a Drucker-Prager criterion.
5. Tool is kinematic; soil does not push back on the tool.
6. DRFT parameters are tuned; not calibrated to a specific soil.
7. No energy equation; dissipation is implicit via friction and numerical flux.

## Mapping to Code

- Mass flux: `_mass_flux_divergence` in `swe_simulator_jax.py`
- mu(I): `_mu_I`
- Friction: `_friction_accel`
- Tool source: `_tool_accel`
- Step update: `step`
