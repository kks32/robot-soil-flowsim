"""
Dynamic trenching experiment that visualizes how the blade displacement
pushes material forward while avalanches propagate along the trench walls.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sand_simulator import SandSimulator


def select_frames(history, count=4):
    """Pick evenly-spaced snapshots from the recorded history."""
    if not history:
        return []
    indices = np.linspace(0, len(history) - 1, count, dtype=int)
    return [history[idx] for idx in indices]


def plot_frames(frames, output_path, flat_height):
    """Render selected frames with tool footprint overlays."""
    if not frames:
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(5 * len(frames), 4))
    for ax, frame in zip(axes, frames):
        im = ax.imshow(
            frame["height"],
            origin="lower",
            cmap=cm.terrain,
            vmin=flat_height - 0.06,
            vmax=flat_height + 0.02,
        )
        if frame["tool_mask"].any():
            ax.contour(
                frame["tool_mask"].astype(float),
                levels=[0.5],
                colors="red",
                linewidths=1.5,
                linestyles="--",
            )
        ax.set_title(f"Step {frame['step']}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Height (m)")
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved dynamic trench visualization to {output_path}")


def plot_3d_frames(frames, grid_x, grid_y, output_path, z_limits):
    """Render 3D surfaces for selected frames."""
    if not frames:
        return

    fig = plt.figure(figsize=(6 * len(frames), 5))
    for idx, frame in enumerate(frames, start=1):
        ax = fig.add_subplot(1, len(frames), idx, projection="3d")
        surf = ax.plot_surface(
            grid_x,
            grid_y,
            frame["height"],
            cmap=cm.terrain,
            linewidth=0,
            antialiased=True,
            vmin=z_limits[0],
            vmax=z_limits[1],
        )
        ax.set_title(f"Step {frame['step']}")
        ax.set_zlim(z_limits[0], z_limits[1])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Height (m)")
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved 3D trench visualization to {output_path}")


def main():
    sim = SandSimulator(grid_size=64, domain_size=1.0, angle_of_repose=29.0)
    flat_height = 0.10
    sim.height[:, :] = flat_height

    print("=" * 70)
    print("DYNAMIC TRENCHING EXPERIMENT")
    print("=" * 70)
    print(f"Flat bed initialized at {flat_height:.2f} m")

    indent_depth = 0.05
    history = sim.push_with_blade(
        x0=0.2,
        y0=0.55,
        x1=0.8,
        y1=0.45,
        indent_depth=indent_depth,
        blade_width=0.12,
        surface_height=flat_height,
        relax_iterations=4,
        record_history=True,
        conserve_mass=False,
    )

    print(f"Recorded {len(history)} intermediate tool states.")

    # Let the trench fully stabilize after the blade exits.
    settle_iters = sim.stabilize(max_iterations=400, tolerance=1e-6, verbose=False)
    final_snapshot = {
        "step": history[-1]["step"] + settle_iters,
        "height": sim.height.copy(),
        "tool_mask": np.zeros_like(sim.tool_mask),
    }
    history.append(final_snapshot)

    frames = select_frames(history, count=4)
    plot_frames(frames, "trenching_dynamic.png", flat_height=flat_height)
    z_limits = (flat_height - indent_depth - 0.01, flat_height + 0.01)
    plot_3d_frames(frames, sim.X, sim.Y, "trenching_dynamic_3d.png", z_limits)

    grad_y, grad_x = np.gradient(sim.height, sim.dx, sim.dx)
    max_slope = np.max(np.sqrt(grad_x**2 + grad_y**2))
    print(f"Final maximum slope: {max_slope:.3f} "
          f"({np.rad2deg(np.arctan(max_slope)):.1f}°)")
    print("=" * 70)


if __name__ == "__main__":
    main()
