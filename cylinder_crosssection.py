"""
Show 2D cross-sections of cylinder collapsing to sand pile.
Plot the profile and show it converging to the angle of repose.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sand_simulator import SandSimulator

print("=" * 70)
print("CYLINDER COLLAPSE - 2D Cross-Section Analysis")
print("=" * 70)

# Create simulator
sim = SandSimulator(grid_size=64, domain_size=1.0, angle_of_repose=29.0)

# Create CYLINDER
center_i, center_j = 32, 32
radius_cells = 8
cylinder_height = 0.15

for i in range(sim.grid_size):
    for j in range(sim.grid_size):
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        if dist < radius_cells:
            sim.height[i, j] = cylinder_height

print(f"\nAngle of repose: 29° (slope = {sim.b_repose:.4f})")
print(f"Cylinder height: {cylinder_height} m")
print(f"Cylinder radius: {radius_cells * sim.dx:.4f} m")
print()

# Capture cross-sections at different iterations
snapshots = []
snapshots_3d = []  # Store full 3D height fields
snapshot_iters = [0, 5, 10, 20, 40, 80, 120]
snapshot_set = set(snapshot_iters)

total_iterations = max(snapshot_iters)
for iter_idx in range(total_iterations + 1):
    if iter_idx in snapshot_set:
        # Extract cross-section through center
        profile = sim.height[center_i, :].copy()
        snapshots.append((iter_idx, profile.copy()))

        # Store full 3D height field
        snapshots_3d.append((iter_idx, sim.height.copy()))

        # Calculate slope on left side of pile
        max_slope = 0
        for ii in range(1, sim.grid_size-1):
            for jj in range(1, sim.grid_size-1):
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    slope = abs((sim.height[ii,jj] - sim.height[ii+di,jj+dj]) / sim.dx)
                    max_slope = max(max_slope, slope)

        print(f"Iter {iter_idx:3d}: max_slope = {max_slope:.4f} ({np.rad2deg(np.arctan(max_slope)):5.1f}°)")

    sim.update_step()

# Continue iterating until the pile is fully stabilized so the final slope
# matches the configured friction angle.
stabilize_iters = sim.stabilize(max_iterations=1000, tolerance=1e-7, verbose=False)
final_iter = total_iterations + stabilize_iters
profile = sim.height[center_i, :].copy()
snapshots.append((final_iter, profile.copy()))
snapshots_3d.append((final_iter, sim.height.copy()))

# Report the stabilized slope that now matches the friction angle.
max_slope = 0
for ii in range(1, sim.grid_size-1):
    for jj in range(1, sim.grid_size-1):
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            slope = abs((sim.height[ii,jj] - sim.height[ii+di,jj+dj]) / sim.dx)
            max_slope = max(max_slope, slope)

print(f"Iter {final_iter:3d} (stabilized): max_slope = {max_slope:.4f} ({np.rad2deg(np.arctan(max_slope)):5.1f}°)")

print()

# Create figure with 3D view and cross-sections
fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

# --- LEFT PANEL: 3D View ---
ax_3d = fig.add_subplot(gs[0, 0], projection='3d')

# Get final (stabilized) state for 3D visualization
final_iter_num, final_height = snapshots_3d[-1]
X, Y = np.meshgrid(sim.x, sim.y)

# Plot 3D surface with colormap
surf = ax_3d.plot_surface(X, Y, final_height, cmap='terrain',
                          edgecolor='none', alpha=0.9, vmin=0, vmax=cylinder_height)

ax_3d.set_xlabel('X (m)', fontsize=11, labelpad=8)
ax_3d.set_ylabel('Y (m)', fontsize=11, labelpad=8)
ax_3d.set_zlabel('Height (m)', fontsize=11, labelpad=8)
ax_3d.set_title(f'3D View: Final State (Iter {final_iter_num})',
                fontsize=13, fontweight='bold', pad=15)
ax_3d.view_init(elev=25, azim=45)
ax_3d.set_zlim(0, cylinder_height * 1.1)

# Add colorbar for 3D plot
cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('Height (m)', fontsize=10)

# --- RIGHT PANEL: Cross-sections ---
ax1 = fig.add_subplot(gs[0, 1])
x_coords = sim.x
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

for idx, (iter_num, profile) in enumerate(snapshots):
    ax1.plot(x_coords, profile, label=f'Iter {iter_num}', 
             linewidth=2, color=colors[idx], alpha=0.8)

# Draw angle of repose reference lines
center_x = 0.5
center_idx = sim.grid_size // 2
pile_top = snapshots[-1][1][center_idx]

# Calculate expected slope lines at angle of repose
dx_side = 0.2  # horizontal distance
dh_side = dx_side * sim.b_repose  # vertical drop

# Left side angle of repose line
ax1.plot([center_x - dx_side, center_x], [pile_top - dh_side, pile_top], 
         'r--', linewidth=2, label=f'Angle of Repose (29°)')

# Right side angle of repose line  
ax1.plot([center_x, center_x + dx_side], [pile_top, pile_top - dh_side], 
         'r--', linewidth=2)

ax1.set_xlabel('X Position (m)', fontsize=11)
ax1.set_ylabel('Height (m)', fontsize=11)
ax1.set_title('Cross-Section Evolution: Cylinder → Sand Pile', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim(-0.01, cylinder_height * 1.1)

plt.savefig('cylinder_crosssection.png', dpi=150, bbox_inches='tight')
print("Saved: cylinder_crosssection.png")

# Additional analysis plot: slope convergence over iterations
fig2, ax = plt.subplots(figsize=(10, 6))

iter_nums = [s[0] for s in snapshots]
max_slopes = []

for iter_num, profile in snapshots:
    max_slope = 0
    for j in range(1, len(profile) - 1):
        slope = abs((profile[j] - profile[j-1]) / sim.dx)
        max_slope = max(max_slope, slope)
    max_slopes.append(max_slope)

ax.plot(iter_nums, max_slopes, 'bo-', linewidth=2, markersize=8, label='Max Slope')
ax.axhline(y=sim.b_repose, color='r', linestyle='--', linewidth=2, 
           label=f'Target: Angle of Repose (29°)')
ax.fill_between(iter_nums, 0, sim.b_repose, alpha=0.2, color='green', 
                label='Stable region')
ax.fill_between(iter_nums, sim.b_repose, max(max_slopes) * 1.1, alpha=0.2, 
                color='red', label='Unstable region')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Maximum Slope (dh/dx)', fontsize=12)
ax.set_title('Convergence to Angle of Repose', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim(0, max(max_slopes) * 1.1)

plt.tight_layout()
plt.savefig('slope_convergence.png', dpi=150, bbox_inches='tight')
print("Saved: slope_convergence.png")

plt.show()

print("\n" + "=" * 70)
print("✓ Cross-section analysis complete!")
final_slope = max_slopes[-1]
final_angle = np.rad2deg(np.arctan(final_slope))
target_angle = np.rad2deg(np.arctan(sim.b_repose))
print(f"  Final max slope: {final_slope:.4f} ({final_angle:.1f}°)")
print(f"  Target (29°):    {sim.b_repose:.4f} ({target_angle:.1f}°)")
print("=" * 70)
