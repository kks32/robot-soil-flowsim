"""
Cylinder collapse comparison: baseline slope-flow vs μ(I) rheology and
dense vs loose packing. Matches the cylinder_crosssection setup and
prints slope evolution for each configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from sand_simulator import SandSimulator


def prepare_sim(phi0, use_muI):
    mu_base = np.tan(np.deg2rad(29.0))
    sim = SandSimulator(
        grid_size=64,
        domain_size=1.0,
        angle_of_repose=29.0,
        use_muI=use_muI,
        mu_s=mu_base,
        mu_2=mu_base + 0.1,
    )
    if use_muI:
        sim.set_initial_phi(phi0)

    center = sim.grid_size // 2
    radius_cells = 8
    cylinder_height = 0.15
    for i in range(sim.grid_size):
        for j in range(sim.grid_size):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist < radius_cells:
                sim.height[i, j] = cylinder_height
    return sim


def run_and_capture(sim, snapshot_iters, label, max_iter=150):
    snapshots = []
    snapshot_set = set(snapshot_iters)
    center_idx = sim.grid_size // 2

    for iter_idx in range(max_iter):
        if iter_idx in snapshot_set:
            snapshots.append((iter_idx, sim.height[center_idx, :].copy()))
        change = sim.update_step()

        max_slope = 0.0
        for ii in range(1, sim.grid_size - 1):
            for jj in range(1, sim.grid_size - 1):
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    slope = abs((sim.height[ii, jj] - sim.height[ii + di, jj + dj]) / sim.dx)
                    max_slope = max(max_slope, slope)
        if iter_idx in snapshot_set:
            print(f"{label} Iter {iter_idx:3d}: max_slope = {max_slope:.4f} ({np.rad2deg(np.arctan(max_slope)):5.1f}°)")

        if change < 1e-6:
            break

    sim.stabilize(max_iterations=300, tolerance=1e-6, verbose=False)
    snapshots.append(("final", sim.height[center_idx, :].copy()))

    final_slope = 0.0
    for ii in range(1, sim.grid_size - 1):
        for jj in range(1, sim.grid_size - 1):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                slope = abs((sim.height[ii, jj] - sim.height[ii + di, jj + dj]) / sim.dx)
                final_slope = max(final_slope, slope)
    final_angle = np.rad2deg(np.arctan(final_slope))
    target_angle = 29.0
    print(f"{label} Final slope: {final_slope:.4f} ({final_angle:.1f}°), target {target_angle:.1f}°")

    return snapshots, final_angle


def plot_comparison(baseline_dense, baseline_loose, mu_dense, mu_loose,
                    baseline_dense_angle, baseline_loose_angle,
                    mu_dense_angle, mu_loose_angle,
                    x_coords, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("Cylinder Collapse: Baseline vs μ(I) for Dense/Loose Sand", fontweight="bold")

    configs = [
        ("Baseline Dense", baseline_dense, baseline_dense_angle, axes[0, 0]),
        ("Baseline Loose", baseline_loose, baseline_loose_angle, axes[1, 0]),
        ("μ(I) Dense", mu_dense, mu_dense_angle, axes[0, 1]),
        ("μ(I) Loose", mu_loose, mu_loose_angle, axes[1, 1]),
    ]

    for title, snapshots, final_angle, ax in configs:
        colors = plt.cm.plasma(np.linspace(0, 1, len(snapshots)))
        for idx, (iter_idx, profile) in enumerate(snapshots):
            ax.plot(x_coords, profile, color=colors[idx], linewidth=2,
                    label=f"Iter {iter_idx}" if iter_idx != "final" else "Final")
        ax.set_title(f"{title}\nFinal slope ≈ {final_angle:.1f}°")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Height (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=150)
    plt.show()


if __name__ == "__main__":
    snapshot_iters = [0, 5, 15, 30, 60, 90]

    baseline_dense = prepare_sim(phi0=0.62, use_muI=False)
    baseline_loose = prepare_sim(phi0=0.56, use_muI=False)
    mu_dense = prepare_sim(phi0=0.62, use_muI=True)
    mu_loose = prepare_sim(phi0=0.56, use_muI=True)

    baseline_dense_snaps, baseline_dense_angle = run_and_capture(baseline_dense, snapshot_iters, "Baseline Dense")
    baseline_loose_snaps, baseline_loose_angle = run_and_capture(baseline_loose, snapshot_iters, "Baseline Loose")
    mu_dense_snaps, mu_dense_angle = run_and_capture(mu_dense, snapshot_iters, "μ(I) Dense")
    mu_loose_snaps, mu_loose_angle = run_and_capture(mu_loose, snapshot_iters, "μ(I) Loose")

    plot_comparison(baseline_dense_snaps, baseline_loose_snaps,
                    mu_dense_snaps, mu_loose_snaps,
                    baseline_dense_angle, baseline_loose_angle,
                    mu_dense_angle, mu_loose_angle,
                    baseline_dense.x, "figs/pile_collapse_muI.png")
