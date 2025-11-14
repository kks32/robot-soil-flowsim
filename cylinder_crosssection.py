"""
Show 2D cross-sections of cylinder collapsing to sand pile.
Plot the profile and show it converging to the angle of repose.
"""

import numpy as np
import matplotlib.pyplot as plt
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
snapshot_iters = [0, 5, 10, 20, 40, 80, 120]

for i in range(max(snapshot_iters) + 1):
    if i in snapshot_iters:
        # Extract cross-section through center
        profile = sim.height[center_i, :].copy()
        snapshots.append((i, profile.copy()))
        
        # Calculate slope on left side of pile
        max_slope = 0
        for ii in range(1, sim.grid_size-1):
            for jj in range(1, sim.grid_size-1):
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    slope = abs((sim.height[ii,jj] - sim.height[ii+di,jj+dj]) / sim.dx)
                    max_slope = max(max_slope, slope)
        
        print(f"Iter {i:3d}: max_slope = {max_slope:.4f} ({np.rad2deg(np.arctan(max_slope)):5.1f}°)")
    
    sim.update_step()

print()

# Create figure with cross-sections
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- TOP PANEL: Cross-sections ---
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

ax1.set_xlabel('X Position (m)', fontsize=12)
ax1.set_ylabel('Height (m)', fontsize=12)
ax1.set_title('Cross-Section Evolution: Cylinder → Sand Pile', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(-0.01, cylinder_height * 1.1)

# --- BOTTOM PANEL: Slope vs Position for final state ---
final_profile = snapshots[-1][1]
slopes = []
positions = []

for j in range(1, len(final_profile) - 1):
    slope = abs((final_profile[j] - final_profile[j-1]) / sim.dx)
    slopes.append(slope)
    positions.append(x_coords[j])

ax2.plot(positions, slopes, 'b-', linewidth=2, label='Actual Slope')
ax2.axhline(y=sim.b_repose, color='r', linestyle='--', linewidth=2, 
            label=f'Angle of Repose = {sim.b_repose:.4f} (29°)')

# Shade region above angle of repose
ax2.fill_between(positions, sim.b_repose, max(slopes) * 1.1, 
                  alpha=0.2, color='red', label='Unstable (slope too steep)')

ax2.set_xlabel('X Position (m)', fontsize=12)
ax2.set_ylabel('Slope (dh/dx)', fontsize=12)
ax2.set_title(f'Final Slope Profile (Iteration {snapshot_iters[-1]})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, max(max(slopes) * 1.1, sim.b_repose * 1.5))

plt.tight_layout()
plt.savefig('/Users/krishna/Downloads/excavate/cylinder_crosssection.png', dpi=150, bbox_inches='tight')
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
plt.savefig('/Users/krishna/Downloads/excavate/slope_convergence.png', dpi=150, bbox_inches='tight')
print("Saved: slope_convergence.png")

plt.show()

print("\n" + "=" * 70)
print("✓ Cross-section analysis complete!")
print(f"  Final max slope: {max_slopes[-1]:.4f} ({np.rad2deg(np.arctan(max_slopes[-1])):.1f}°)")
print(f"  Target (29°):    {sim.b_repose:.4f}")
print("=" * 70)
