from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from swe_simulator_jax import SimParams, init_flat, make_grid, rollout


def _parse_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_tool_sequence(
    params: SimParams,
    x0: float,
    x1: float,
    y: float,
    z: float,
    speed: float,
    settle_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    dist = abs(x1 - x0)
    speed_safe = max(speed, 1e-6)
    travel_time = dist / speed_safe
    steps = max(int(np.ceil(travel_time / params.dt)), 2)
    xs = jnp.linspace(x0, x1, steps)
    ys = jnp.full_like(xs, y)
    zs = jnp.full_like(xs, z)
    pos = jnp.stack([xs, ys, zs], axis=-1)

    signed_speed = (x1 - x0) / (steps * params.dt)
    vel = jnp.stack([jnp.full((steps,), signed_speed), jnp.zeros((steps,))], axis=-1)

    if settle_steps > 0:
        pos_settle = jnp.repeat(pos[-1][None, :], settle_steps, axis=0)
        vel_settle = jnp.zeros((settle_steps, 2))
        pos = jnp.concatenate([pos, pos_settle], axis=0)
        vel = jnp.concatenate([vel, vel_settle], axis=0)

    return pos, vel, steps


def run_case(
    params: SimParams,
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    base_state,
    trench_depth: float,
    speed: float,
    x0: float,
    x1: float,
    y: float,
    settle_steps: int,
    rollout_fn,
) -> tuple[np.ndarray, dict]:
    tool_z = float(base_state.h.max()) - trench_depth
    tool_pos, tool_vel, steps = make_tool_sequence(
        params,
        x0=x0,
        x1=x1,
        y=y,
        z=tool_z,
        speed=speed,
        settle_steps=settle_steps,
    )

    final_state, _ = rollout_fn(base_state, tool_pos, tool_vel, grid_x, grid_y, params)
    h_final = np.array(final_state.h)

    grad_y, grad_x = np.gradient(h_final, params.dx, params.dx)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    max_slope = float(np.max(slope))
    max_angle = float(np.rad2deg(np.arctan(max_slope)))

    mass_init = float(np.sum(base_state.h) * params.dx * params.dx * params.rho_bulk)
    mass_final = float(np.sum(h_final) * params.dx * params.dx * params.rho_bulk)

    metrics = {
        "steps": steps,
        "mass_change_pct": (mass_final - mass_init) / mass_init * 100.0,
        "max_slope": max_slope,
        "max_angle_deg": max_angle,
    }
    return h_final, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX SWE trench sweep.")
    parser.add_argument("--depths", default="0.03,0.05,0.07", help="comma-separated depths (m)")
    parser.add_argument("--speeds", default="0.15,0.30", help="comma-separated tool speeds (m/s)")
    parser.add_argument("--grid", type=int, default=96, help="grid resolution")
    parser.add_argument("--dt", type=float, default=0.002, help="time step (s)")
    parser.add_argument("--settle", type=int, default=300, help="settle steps after cutting")
    parser.add_argument("--fast", action="store_true", help="quick smoke test")
    args = parser.parse_args()

    if args.fast:
        depths = [0.04]
        speeds = [0.3]
        grid_size = 64
        settle_steps = 100
    else:
        depths = _parse_list(args.depths)
        speeds = _parse_list(args.speeds)
        grid_size = args.grid
        settle_steps = args.settle

    domain = 1.0
    dx = domain / (grid_size - 1)
    params = SimParams(
        nx=grid_size,
        ny=grid_size,
        dx=dx,
        dt=args.dt,
        mu_s=float(np.tan(np.deg2rad(29.0))),
        mu_2=float(np.tan(np.deg2rad(33.0))),
        tool_width=0.12,
        tool_length=0.08,
        tool_drag=5.0e4,
        tool_splash=1.2,
        tool_softness=0.01,
    )

    grid_x, grid_y = make_grid(params)
    h0 = 0.10
    base_state = init_flat(params, height=h0)
    h_init = np.array(base_state.h)
    rollout_fn = jax.jit(rollout, static_argnames=("params",))

    results = {}
    for depth in depths:
        for speed in speeds:
            h_final, metrics = run_case(
                params,
                grid_x,
                grid_y,
                base_state,
                trench_depth=depth,
                speed=speed,
                x0=0.2,
                x1=0.8,
                y=0.5,
                settle_steps=settle_steps,
                rollout_fn=rollout_fn,
            )
            results[(depth, speed)] = (h_final, metrics)
            print(
                f"depth={depth:.3f}m speed={speed:.2f}m/s steps={metrics['steps']} "
                f"massΔ={metrics['mass_change_pct']:.3f}% "
                f"max_slope={metrics['max_slope']:.3f} ({metrics['max_angle_deg']:.1f}°)"
            )

    rows = len(depths)
    cols = len(speeds)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    x_axis = np.linspace(0.0, domain, grid_size)
    mid = grid_size // 2

    for r, depth in enumerate(depths):
        for c, speed in enumerate(speeds):
            h_final, metrics = results[(depth, speed)]
            ax = axes[r][c]
            ax.plot(x_axis, h_init[mid, :], color="gray", linestyle="--", linewidth=1.0, label="Initial")
            ax.plot(x_axis, h_final[mid, :], color="tab:blue", linewidth=1.6, label="Final")
            ax.set_title(f"d={depth:.02f}m, v={speed:.2f}m/s")
            ax.set_ylim(0.0, h0 * 1.05)
            ax.grid(alpha=0.25)
            if r == rows - 1:
                ax.set_xlabel("x (m)")
            if c == 0:
                ax.set_ylabel("Height (m)")
            ax.text(
                0.02,
                h0 * 0.97,
                f"Δm={metrics['mass_change_pct']:.2f}%\n"
                f"θmax={metrics['max_angle_deg']:.1f}°",
                fontsize=8.5,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    fig.tight_layout()
    fig.savefig("swe_trench_jax_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved swe_trench_jax_sweep.png")

    fig2, axes2 = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for r, depth in enumerate(depths):
        for c, speed in enumerate(speeds):
            h_final, _ = results[(depth, speed)]
            ax = axes2[r][c]
            im = ax.imshow(h_final, origin="lower", cmap="terrain", vmin=h0 - 0.08, vmax=h0 + 0.02)
            ax.set_title(f"d={depth:.02f}m, v={speed:.2f}m/s")
            ax.set_xticks([])
            ax.set_yticks([])
            fig2.colorbar(im, ax=ax, shrink=0.8)

    fig2.tight_layout()
    fig2.savefig("swe_trench_jax_maps.png", dpi=150, bbox_inches="tight")
    print("Saved swe_trench_jax_maps.png")


if __name__ == "__main__":
    main()
