import argparse
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from sand_simulator import SandSimulator
from swe_simulator_jax import SimParams, init_flat, make_grid, rollout


def run_trench_reference(
    depth: float,
    blade_width: float,
    grid_size: int,
    domain: float,
    flat_height: float,
):
    sim = SandSimulator(grid_size=grid_size, domain_size=domain, angle_of_repose=29.0)
    sim.height[:, :] = flat_height
    sim.push_with_blade(
        x0=0.2,
        y0=0.5,
        x1=0.8,
        y1=0.5,
        indent_depth=depth,
        blade_width=blade_width,
        surface_height=flat_height,
        relax_iterations=0,
        conserve_mass=True,
    )
    sim.stabilize(max_iterations=400, tolerance=1e-6, verbose=False)
    return sim.height.copy(), sim.x


def run_trench_jax(
    depth: float,
    blade_width: float,
    speed: float,
    grid_size: int,
    domain: float,
    flat_height: float,
    tool_drag: float,
    tool_splash: float,
    settle_steps: int,
):
    dx = domain / (grid_size - 1)
    mu_s = math.tan(math.radians(29.0))
    mu_2 = math.tan(math.radians(33.0))
    params = SimParams(
        nx=grid_size,
        ny=grid_size,
        dx=dx,
        dt=0.002,
        mu_s=mu_s,
        mu_2=mu_2,
        tool_width=blade_width,
        tool_length=0.08,
        tool_drag=tool_drag,
        tool_splash=tool_splash,
    )
    grid_x, grid_y = make_grid(params)
    base_state = init_flat(params, height=flat_height)

    tool_z = flat_height - depth
    travel_dist = 0.6
    travel_time = travel_dist / max(speed, 1e-6)
    steps = max(int(np.ceil(travel_time / params.dt)), 2)

    xs = jnp.linspace(0.2, 0.8, steps)
    ys = jnp.full_like(xs, 0.5)
    zs = jnp.full_like(xs, tool_z)
    tool_pos = jnp.stack([xs, ys, zs], axis=-1)
    tool_vel = jnp.stack(
        [jnp.full((steps,), travel_dist / (steps * params.dt)), jnp.zeros((steps,))],
        axis=-1,
    )

    pos_settle = jnp.repeat(tool_pos[-1][None, :], settle_steps, axis=0)
    vel_settle = jnp.zeros((settle_steps, 2))
    tool_pos = jnp.concatenate([tool_pos, pos_settle], axis=0)
    tool_vel = jnp.concatenate([tool_vel, vel_settle], axis=0)

    rollout_fn = jax.jit(rollout, static_argnames=("params",))
    final_state, _ = rollout_fn(base_state, tool_pos, tool_vel, grid_x, grid_y, params)
    return np.array(final_state.h), params


def metrics(h: np.ndarray, flat_height: float, dx: float):
    trench_depth = float(flat_height - np.min(h))
    berm_height = float(np.max(h) - flat_height)
    grad_y, grad_x = np.gradient(h, dx, dx)
    max_slope = float(np.max(np.sqrt(grad_x**2 + grad_y**2)))
    max_angle = float(np.rad2deg(np.arctan(max_slope)))
    return trench_depth, berm_height, max_angle


def _parse_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(b), 1e-6)
    return abs(a - b) / denom


def main():
    parser = argparse.ArgumentParser(description="Verify SWE JAX vs trench.py baseline.")
    parser.add_argument("--depth", type=float, default=0.05, help="trench depth (m)")
    parser.add_argument("--speeds", default="0.15,0.30,0.45", help="comma-separated tool speeds (m/s)")
    parser.add_argument("--grid", type=int, default=64, help="grid resolution")
    parser.add_argument("--blade-width", type=float, default=0.15, help="blade width (m)")
    parser.add_argument("--settle", type=int, default=300, help="settle steps after tool motion")
    parser.add_argument("--tool-drag", type=float, default=None, help="override tool drag coefficient")
    parser.add_argument("--tool-splash", type=float, default=None, help="override tool splash coefficient")
    parser.add_argument("--calibrate-drag", action="store_true", help="grid-search tool drag to match reference")
    parser.add_argument(
        "--drag-values",
        default="20000,30000,40000,50000,60000,80000",
        help="comma-separated tool drag values for calibration",
    )
    args = parser.parse_args()

    speeds = _parse_list(args.speeds)
    if not speeds:
        raise ValueError("At least one speed must be provided.")

    flat_height = 0.10
    domain = 1.0
    ref_h, x_axis = run_trench_reference(
        depth=args.depth,
        blade_width=args.blade_width,
        grid_size=args.grid,
        domain=domain,
        flat_height=flat_height,
    )
    ref_params_dx = domain / (args.grid - 1)

    ref_depth, ref_berm, ref_angle = metrics(ref_h, flat_height, ref_params_dx)

    print("Reference (trench.py / SandSimulator)")
    print(f"  depth={ref_depth:.3f} m  berm={ref_berm:.3f} m  max_slope={ref_angle:.1f}°")

    if args.calibrate_drag:
        drag_values = _parse_list(args.drag_values)
        if not drag_values:
            raise ValueError("At least one drag value must be provided for calibration.")

        tool_splash = 0.0 if args.tool_splash is None else args.tool_splash
        best = None
        slow_speed = min(speeds)
        for drag in drag_values:
            jax_h, params = run_trench_jax(
                depth=args.depth,
                blade_width=args.blade_width,
                speed=slow_speed,
                grid_size=args.grid,
                domain=domain,
                flat_height=flat_height,
                tool_drag=drag,
                tool_splash=tool_splash,
                settle_steps=args.settle,
            )
            jax_depth, jax_berm, jax_angle = metrics(jax_h, flat_height, params.dx)
            score = (
                _relative_error(jax_depth, ref_depth)
                + _relative_error(jax_berm, ref_berm)
                + _relative_error(jax_angle, ref_angle)
            )
            print(
                f"calib drag={drag:.0f} v={slow_speed:.2f} "
                f"depth={jax_depth:.3f} berm={jax_berm:.3f} slope={jax_angle:.1f} "
                f"score={score:.3f}"
            )
            if best is None or score < best["score"]:
                best = {
                    "drag": drag,
                    "score": score,
                    "depth": jax_depth,
                    "berm": jax_berm,
                    "angle": jax_angle,
                }

        if best is None:
            raise RuntimeError("Calibration failed to evaluate any drag values.")

        print("Best drag fit")
        print(
            f"  drag={best['drag']:.0f} score={best['score']:.3f} "
            f"depth={best['depth']:.3f} berm={best['berm']:.3f} slope={best['angle']:.1f}°"
        )

    tool_drag = 5.0e4 if args.tool_drag is None else args.tool_drag
    tool_splash = 1.2 if args.tool_splash is None else args.tool_splash
    if args.calibrate_drag and best is not None:
        tool_drag = best["drag"]

    mid = args.grid // 2
    fig, axes = plt.subplots(
        1,
        len(speeds),
        figsize=(5 * len(speeds), 4),
        squeeze=False,
    )
    for idx, speed in enumerate(speeds):
        jax_h, params = run_trench_jax(
            depth=args.depth,
            blade_width=args.blade_width,
            speed=speed,
            grid_size=args.grid,
            domain=domain,
            flat_height=flat_height,
            tool_drag=tool_drag,
            tool_splash=tool_splash,
            settle_steps=args.settle,
        )
        jax_depth, jax_berm, jax_angle = metrics(jax_h, flat_height, params.dx)
        print(f"JAX SWE @ v={speed:.2f} m/s")
        print(f"  depth={jax_depth:.3f} m  berm={jax_berm:.3f} m  max_slope={jax_angle:.1f}°")
        print("  Differences (JAX - Ref)")
        print(
            f"    depth={jax_depth - ref_depth:+.3f} m  "
            f"berm={jax_berm - ref_berm:+.3f} m  "
            f"max_slope={jax_angle - ref_angle:+.1f}°"
        )

        ax = axes[0][idx]
        ax.plot(x_axis, ref_h[mid, :], label="Reference", color="black", linewidth=1.2)
        ax.plot(x_axis, jax_h[mid, :], label=f"JAX v={speed:.2f}", color="tab:blue")
        ax.axhline(flat_height, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("x (m)")
        if idx == 0:
            ax.set_ylabel("Height (m)")
        ax.set_title(f"v={speed:.2f} m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Centerline Profile Comparison", fontsize=12)
    fig.tight_layout()
    output = "verify_trench_compare_speeds.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
