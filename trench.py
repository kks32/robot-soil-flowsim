"""
Simple test: Flat bed, create trench (negative depth), see sides avalanche in.
Shows BEFORE and AFTER profiles.
Uses wedge-based force limits to plan multiple passes to reach target depth.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sand_simulator import SandSimulator


def wedge_force(
    theta,
    depth,
    width,
    gamma,
    phi,
    delta,
    cohesion=0.0,
    surcharge=0.0,
):
    """
    Limit equilibrium wedge force (Hettiaratchi–Reece style).

    Parameters are in SI units, angles in radians.
    theta    : failure plane angle to the horizontal
    depth    : cut depth (m)
    width    : blade width (m)
    gamma    : unit weight of soil (kN/m³)
    phi      : soil friction angle (rad)
    delta    : soil–tool friction angle (rad)
    cohesion : cohesion (kPa)
    surcharge: surcharge on ground surface (kPa)

    Returns:
        Draft force F (kN per meter out of plane).
    """
    # Geometry of triangular wedge
    runout = depth / np.tan(theta)
    area = 0.5 * depth * runout

    # Forces acting on the wedge
    weight = gamma * width * area  # kN
    surcharge_force = surcharge * width * runout  # kN
    failure_plane_length = depth / np.sin(theta)
    adhesion_force = cohesion * width * failure_plane_length  # kN

    # Eq. 4 form (Swick & Perumpral, Rathore et al. 2025): 
    # F = [ W*sin(theta+phi) + ... ] / cos(theta+delta+phi)
    # Note: The physical "dead zone" (Rathore et al.) effectively increases the rake angle, 
    # but for this limit equilibrium calculation, we use the tool geometry directly.
    cos_term = np.cos(theta + delta + phi)
    
    # Avoid singularity when cos(theta+delta+phi) -> 0 (Lock-up condition)
    # In the denominator, this drives force -> infinity.
    eps = 1e-3
    cos_term = np.sign(cos_term) * np.maximum(np.abs(cos_term), eps)

    # Corrected formula: cos_term is in the denominator.
    force = (
        (weight + surcharge_force) * np.sin(theta + phi) + adhesion_force * np.cos(phi)
    ) / cos_term

    # Passive-wedge lower bound (Coulomb earth pressure style) to avoid vanishing forces
    kp = np.tan(np.pi / 4 + phi / 2) ** 2
    coulomb_force = 0.5 * gamma * width * (depth**2) * kp / np.maximum(np.cos(delta), eps)

    return np.maximum(np.abs(force), coulomb_force)  # magnitude


def optimal_wedge(
    depth,
    width,
    gamma,
    phi,
    delta,
    cohesion=0.0,
    surcharge=0.0,
    theta_bounds=(np.deg2rad(20), np.deg2rad(70)),
    n_samples=400,
):
    """
    Sample theta to find the minimum draft force configuration.
    """
    thetas = np.linspace(theta_bounds[0], theta_bounds[1], n_samples)
    forces = wedge_force(
        thetas, depth, width, gamma, phi, delta, cohesion=cohesion, surcharge=surcharge
    )
    idx = np.argmin(forces)
    return thetas[idx], forces[idx], thetas, forces


def max_depth_for_force(
    force_limit,
    width,
    gamma,
    phi,
    delta,
    cohesion=0.0,
    surcharge=0.0,
    depth_max=0.2,
    theta_bounds=(np.deg2rad(20), np.deg2rad(70)),
    safety_factor=1.5,
):
    """
    Binary search the maximum incremental depth whose minimum draft (times safety factor) ≤ force_limit.
    """
    lo, hi = 0.0, depth_max
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        theta_mid, force_mid, _, _ = optimal_wedge(
            depth=mid,
            width=width,
            gamma=gamma,
            phi=phi,
            delta=delta,
            cohesion=cohesion,
            surcharge=surcharge,
            theta_bounds=theta_bounds,
        )
        # Check against limit with safety factor for oscillations
        if force_mid * safety_factor <= force_limit:
            lo = mid
        else:
            hi = mid
    return lo


def plan_force_limited_passes(
    target_depth,
    force_limit,
    width,
    gamma,
    phi,
    delta,
    cohesion=0.0,
    surcharge=0.0,
    theta_bounds=(np.deg2rad(20), np.deg2rad(70)),
    safety_factor=1.5,
):
    """
    Plan cumulative depths so each incremental cut stays under force_limit.
    Returns a list of dicts with keys: increment, cum_depth, theta, force.
    
    safety_factor: Multiplier (default 1.5) to account for force oscillations 
                   (cyclic shear band formation) described in Rathore et al. (2025).
    """
    if target_depth <= 0:
        return []

    max_inc = max_depth_for_force(
        force_limit,
        width=width,
        gamma=gamma,
        phi=phi,
        delta=delta,
        cohesion=cohesion,
        surcharge=surcharge,
        depth_max=target_depth,
        theta_bounds=theta_bounds,
        safety_factor=safety_factor,
    )
    if max_inc <= 1e-6:
        raise ValueError("Force limit too low to cut any depth.")

    passes = []
    depth_done = 0.0
    while depth_done + 1e-9 < target_depth:
        inc = min(max_inc, target_depth - depth_done)
        theta_inc, force_inc, _, _ = optimal_wedge(
            depth=inc,
            width=width,
            gamma=gamma,
            phi=phi,
            delta=delta,
            cohesion=cohesion,
            surcharge=surcharge,
            theta_bounds=theta_bounds,
        )
        depth_done += inc
        passes.append(
            {
                "increment": inc,
                "cum_depth": depth_done,
                "theta": theta_inc,
                "force": force_inc,
            }
        )
    return passes


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

# Soil parameters informed by Rathore et al. (2025) (Quartz Sand)
# Note: Phi=40 deg for Quartz sand (vs 30 deg for Glass beads in 2022 paper).
phi_deg = 40.0  # critical state friction angle under plane strain
# Note: Delta=23 deg (approx) observed experimentally, significantly higher 
# than empirical estimates (like Grisso's ~10 deg).
delta_deg = 23.0  # sand–steel interface friction
gamma = 15.0  # kN/m³, bulk unit weight for loose-to-medium packing
cohesion = 0.0  # kPa, dry sand assumed cohesionless
surcharge = 0.0  # kPa, level ground
target_depth = 0.10  # desired final trench depth (m)
force_limit = 0.020  # kN/m allowable draft per meter (≈20 N/m)
blade_width = 0.15
THETA_BOUNDS = (np.deg2rad(20), np.deg2rad(70))  # avoid cos(θ+δ+φ) ≈ 0 singularity and shallow wedges
STABILIZE_EACH_PASS = True  # let avalanche settle between passes
SAFETY_FACTOR = 1.5 # Account for oscillatory peaks

max_inc = max_depth_for_force(
    force_limit,
    width=blade_width,
    gamma=gamma,
    phi=np.deg2rad(phi_deg),
    delta=np.deg2rad(delta_deg),
    cohesion=cohesion,
    surcharge=surcharge,
    depth_max=target_depth,
    theta_bounds=THETA_BOUNDS,
    safety_factor=SAFETY_FACTOR,
)
theta_cap, force_cap, _, _ = optimal_wedge(
    depth=max_inc,
    width=blade_width,
    gamma=gamma,
    phi=np.deg2rad(phi_deg),
    delta=np.deg2rad(delta_deg),
    cohesion=cohesion,
    surcharge=surcharge,
    theta_bounds=THETA_BOUNDS,
)

print("Force-limited cutting plan (limit-equilibrium wedge):")
print(
    f"  Max depth per pass under force cap: {max_inc:.3f} m "
    f"(F≈{force_cap:.4f} kN/m @ θ≈{np.rad2deg(theta_cap):.1f}°)"
)
pass_plan = plan_force_limited_passes(
    target_depth=target_depth,
    force_limit=force_limit,
    width=blade_width,
    gamma=gamma,
    phi=np.deg2rad(phi_deg),
    delta=np.deg2rad(delta_deg),
    cohesion=cohesion,
    surcharge=surcharge,
    theta_bounds=THETA_BOUNDS,
)
theta_overlay, force_overlay, _, _ = optimal_wedge(
    depth=target_depth,
    width=blade_width,
    gamma=gamma,
    phi=np.deg2rad(phi_deg),
    delta=np.deg2rad(delta_deg),
    cohesion=cohesion,
    surcharge=surcharge,
    theta_bounds=THETA_BOUNDS,
)
print(
    f"  Min draft for full depth {target_depth:.3f} m: "
    f"{force_overlay:.4f} kN/m @ θ≈{np.rad2deg(theta_overlay):.1f}°"
)
if force_overlay > force_limit:
    print("  → Exceeds force limit; multiple passes required.")
print()
for idx, p in enumerate(pass_plan, 1):
    print(
        f"  Pass {idx}: Δd={p['increment']:.3f} m, cumulative d={p['cum_depth']:.3f} m, "
        f"θ≈{np.rad2deg(p['theta']):.1f}°, F≈{p['force']:.4f} kN/m"
    )

# Create figure
fig = plt.figure(figsize=(16, 8))

# Step 1: Flat bed
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(sim.X, sim.Y, sim.height, cmap=cm.terrain, alpha=0.9)
ax1.set_title('Step 1: Flat Bed', fontsize=12, fontweight='bold')
ax1.set_zlim(0, flat_height * 1.2)
ax1.view_init(elev=25, azim=45)

# Step 2: Create trench in multiple passes (force-limited)
print("[Step 2] Creating trench (digging down with force cap)...")
print("  Path: (0.2, 0.5) → (0.8, 0.5)")
print(f"  Target depth: {target_depth:.3f}m, force limit: {force_limit:.4f} kN/m")
pass_profiles = []
pass_depths = []
for idx, p in enumerate(pass_plan, 1):
    print(f"    Executing pass {idx} to cumulative depth {p['cum_depth']:.3f} m")
    sim.push_with_blade(
        x0=0.2,
        y0=0.5,
        x1=0.8,
        y1=0.5,
        indent_depth=p["cum_depth"],
        blade_width=blade_width,
        surface_height=flat_height,
        relax_iterations=0,
        conserve_mass=False,
    )
    if STABILIZE_EACH_PASS:
        iters_pass = sim.stabilize(max_iterations=200, tolerance=1e-6, verbose=False)
        print(f"      Stabilized pass {idx} in {iters_pass} iterations")
    pass_profiles.append(sim.height[center_i := sim.grid_size // 2, :].copy())
    pass_depths.append(p["cum_depth"])

# Save profile BEFORE stabilization (after last pass)
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

# Step 3: Final stabilize (in case last pass left any steep faces)
print("[Step 3] Final stabilization (sides avalanche into trench if needed)...")
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

# Overlay intermediate pass profiles to show multi-pass progression
for prof, depth in zip(pass_profiles, pass_depths):
    ax4.plot(
        sim.x,
        prof,
        linestyle=':',
        linewidth=1.2,
        label=f'Pass @ d={depth:.3f} m',
        alpha=0.8,
    )

# Overlay predicted wedge failure planes from limit equilibrium
wedge_runout = target_depth / np.tan(theta_overlay)
center_x = 0.5
left_wedge_x = np.clip([center_x - wedge_runout, center_x], 0, sim.domain_size)
right_wedge_x = np.clip([center_x, center_x + wedge_runout], 0, sim.domain_size)
wedge_z = [flat_height - target_depth, flat_height]
ax4.plot(left_wedge_x, wedge_z, color='purple', linestyle='-.', linewidth=1.8,
         label=f'Wedge plane θ*={np.rad2deg(theta_overlay):.1f}°')
ax4.plot(right_wedge_x, wedge_z, color='purple', linestyle='-.', linewidth=1.8)

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
last_pass = pass_plan[-1] if pass_plan else {"force": 0.0, "theta": 0.0}
ax4.text(
    0.02,
    flat_height * 1.04,
    "Force limit: {:.4f} kN/m\n"
    "Last pass F≈{:.4f} kN/m @ θ≈{:.1f}°\n"
    "γ={:.1f} kN/m³, φ={:.0f}°, δ={:.0f}°".format(
        force_limit,
        last_pass["force"],
        np.rad2deg(last_pass["theta"]),
        gamma,
        phi_deg,
        delta_deg,
    ),
    fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
)

plt.suptitle('Trench Formation: Flat Bed → Cut → Avalanche', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('trench.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: trench.png")

# Force–depth curve for reference
depth_grid = np.linspace(0.01, target_depth, 40)
force_grid = []
theta_grid = []
for d in depth_grid:
    th_d, f_d, _, _ = optimal_wedge(
        depth=d,
        width=blade_width,
        gamma=gamma,
        phi=np.deg2rad(phi_deg),
        delta=np.deg2rad(delta_deg),
        cohesion=cohesion,
        surcharge=surcharge,
        theta_bounds=THETA_BOUNDS,
    )
    force_grid.append(f_d)
    theta_grid.append(th_d)

plt.figure(figsize=(6, 4))
plt.plot(depth_grid, force_grid, label='Min draft vs depth')
plt.axhline(force_limit, color='red', linestyle='--', label='Force limit')
plt.scatter(pass_depths, [p["force"] for p in pass_plan], color='purple', label='Pass points')
plt.xlabel('Depth (m)')
plt.ylabel('Draft (kN/m)')
plt.title('Force Limit Plan (Wedge Model)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trench_force_plan.png', dpi=150, bbox_inches='tight')
print("✓ Saved: trench_force_plan.png")

plt.show()

print("\n" + "=" * 70)
print("✓ TEST COMPLETE")
print("=" * 70)
