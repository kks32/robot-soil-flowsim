"""
Simple test: Flat bed, create trench (negative depth), see sides avalanche in.
Shows BEFORE and AFTER profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sand_simulator import SandSimulator

print("=" * 70)
print("SIMPLE TRENCH TEST: Flat Bed → Trench → Avalanche")
print("=" * 70)

# Create simulator
sim = SandSimulator(grid_size=64, domain_size=1.0, angle_of_repose=29.0)

# Start with FLAT BED at uniform height
flat_height = 0.10
sim.height[:, :] = flat_height

print(f"\nStarting with flat bed at height: {flat_height}m")
print(f"Angle of repose: 29°")
print()

# Create figure
fig = plt.figure(figsize=(16, 8))

# Step 1: Flat bed
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(sim.X, sim.Y, sim.height, cmap=cm.terrain, alpha=0.9)
ax1.set_title('Step 1: Flat Bed', fontsize=12, fontweight='bold')
ax1.set_zlim(0, flat_height * 1.2)
ax1.view_init(elev=25, azim=45)

# Step 2: Create trench (NEGATIVE - dig down!)
print("[Step 2] Creating trench (digging down)...")
print("  Path: (0.2, 0.5) → (0.8, 0.5)")
print("  Indent depth: 0.05m (NEGATIVE)")

# The blade creates a DEPRESSION
sim.push_with_blade(x0=0.2, y0=0.5, x1=0.8, y1=0.5,
                    indent_depth=0.05, blade_width=0.15)

# Save profile BEFORE stabilization
center_i = sim.grid_size // 2
profile_before = sim.height[center_i, :].copy()

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(sim.X, sim.Y, sim.height, cmap=cm.terrain, alpha=0.9)
ax2.set_title('Step 2: Trench Cut (Unstable Walls)', fontsize=12, fontweight='bold')
ax2.set_zlim(0, flat_height * 1.2)
ax2.view_init(elev=25, azim=45)

# Check slopes
max_slope_before = 0
for i in range(1, sim.grid_size-1):
    for j in range(1, sim.grid_size-1):
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            slope = abs((sim.height[i,j] - sim.height[i+di,j+dj]) / sim.dx)
            max_slope_before = max(max_slope_before, slope)

print(f"  Max slope after cutting: {max_slope_before:.4f} ({np.rad2deg(np.arctan(max_slope_before)):.1f}°)")
print(f"  Angle of repose: {sim.b_repose:.4f} (29.0°)")
print(f"  → Sides are TOO STEEP, will avalanche into trench!")
print()

# Step 3: Stabilize (sides avalanche in)
print("[Step 3] Stabilizing (sides avalanche into trench)...")
iters = sim.stabilize(max_iterations=300, tolerance=1e-6, verbose=False)
print(f"  Converged in {iters} iterations")

# Save profile AFTER stabilization
profile_after = sim.height[center_i, :].copy()

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.plot_surface(sim.X, sim.Y, sim.height, cmap=cm.terrain, alpha=0.9)
ax3.set_title(f'Step 3: After Avalanche ({iters} iters)', fontsize=12, fontweight='bold')
ax3.set_zlim(0, flat_height * 1.2)
ax3.view_init(elev=25, azim=45)

# Check final slopes
max_slope_after = 0
for i in range(1, sim.grid_size-1):
    for j in range(1, sim.grid_size-1):
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            slope = abs((sim.height[i,j] - sim.height[i+di,j+dj]) / sim.dx)
            max_slope_after = max(max_slope_after, slope)

print(f"  Final max slope: {max_slope_after:.4f} ({np.rad2deg(np.arctan(max_slope_after)):.1f}°)")
print(f"  Target: {sim.b_repose:.4f} (29.0°)")

if max_slope_after <= sim.b_repose + 0.05:
    print("  ✓ Trench sides stabilized to angle of repose!")

# Step 4: Cross-section view - BEFORE and AFTER
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(sim.x, profile_before, 'r--', linewidth=2, label='After cutting (unstable)', alpha=0.7)
ax4.plot(sim.x, profile_after, 'b-', linewidth=2, label='After avalanche (stable)')
ax4.axhline(y=flat_height, color='gray', linestyle=':', label='Original flat surface', alpha=0.5, linewidth=1.5)

# Shade the material that fell into the trench
ax4.fill_between(sim.x, profile_before, profile_after, 
                  where=(profile_after > profile_before), 
                  alpha=0.3, color='orange', label='Sand that avalanched in')

ax4.set_xlabel('X position (m)', fontsize=11)
ax4.set_ylabel('Height (m)', fontsize=11)
ax4.set_title('Cross-Section: Before vs After Stabilization', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9, loc='upper right')
ax4.set_ylim(0, flat_height * 1.1)

plt.suptitle('Trench Formation: Flat Bed → Cut → Avalanche', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('test_simple_trench.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: test_simple_trench.png")

plt.show()

print("\n" + "=" * 70)
print("✓ TEST COMPLETE")
print("=" * 70)
