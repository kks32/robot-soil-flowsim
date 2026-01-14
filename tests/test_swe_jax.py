import jax
import jax.numpy as jnp
import numpy as np

from swe_simulator_jax import SimParams, State, init_flat, make_grid, rollout, _mu_I


def test_mass_conservation_no_tool():
    params = SimParams(nx=32, ny=32, dx=1.0 / 31, dt=0.002)
    state0 = init_flat(params, height=0.1)
    grid_x, grid_y = make_grid(params)

    steps = 20
    tool_pos = jnp.stack(
        [
            jnp.full((steps,), 0.5),
            jnp.full((steps,), 0.5),
            jnp.full((steps,), 1.0),
        ],
        axis=-1,
    )
    tool_vel = jnp.zeros((steps, 2))
    rollout_fn = jax.jit(rollout, static_argnames=("params",))

    final_state, _ = rollout_fn(state0, tool_pos, tool_vel, grid_x, grid_y, params)
    mass0 = float(np.sum(np.array(state0.h)))
    mass1 = float(np.sum(np.array(final_state.h)))
    assert abs(mass1 - mass0) / mass0 < 1e-6


def test_mu_I_bounds():
    params = SimParams(nx=16, ny=16, dx=1.0 / 15, dt=0.002)
    h = jnp.full((params.ny, params.nx), 0.1)
    u = jnp.full_like(h, 0.2)
    v = jnp.full_like(h, -0.1)
    mu = _mu_I(u, v, h, params)
    assert float(jnp.min(mu)) >= params.mu_s - 1e-3
    assert float(jnp.max(mu)) <= params.mu_2 + 1e-3


def test_steep_slope_relaxes():
    params = SimParams(nx=32, ny=32, dx=1.0 / 31, dt=0.002)
    h = jnp.full((params.ny, params.nx), 0.1)
    h = h.at[:, : params.nx // 2].add(0.05)
    state0 = State(h=h, hu=jnp.zeros_like(h), hv=jnp.zeros_like(h))
    grid_x, grid_y = make_grid(params)

    steps = 60
    tool_pos = jnp.stack(
        [
            jnp.full((steps,), 0.5),
            jnp.full((steps,), 0.5),
            jnp.full((steps,), 1.0),
        ],
        axis=-1,
    )
    tool_vel = jnp.zeros((steps, 2))
    rollout_fn = jax.jit(rollout, static_argnames=("params",))

    grad_y0, grad_x0 = jnp.gradient(h, params.dx, params.dx)
    max_slope0 = float(jnp.max(jnp.sqrt(grad_x0**2 + grad_y0**2)))

    final_state, _ = rollout_fn(state0, tool_pos, tool_vel, grid_x, grid_y, params)
    grad_y1, grad_x1 = jnp.gradient(final_state.h, params.dx, params.dx)
    max_slope1 = float(jnp.max(jnp.sqrt(grad_x1**2 + grad_y1**2)))

    assert max_slope1 < max_slope0
