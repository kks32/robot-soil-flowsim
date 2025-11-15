"""
Dynamic Trenching Comparison: Force Control vs Displacement Control

Shows the difference between:
1. Displacement control: "Dig 50mm deep" (what we've been doing)
2. Force control: "Push with 1000N normal force" (Bekker-Wong)

Key insight: Same force produces different sinkage in different materials!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from force_controlled_simulator import ForceControlledSimulator


def compare_control_modes():
    """Compare force-controlled vs displacement-controlled trenching."""

    print("Dynamic Trenching: Force vs Displacement Control")
    print("=" * 60)

    # Create THREE simulators:
    # 1. Loose sand with force control
    # 2. Dense sand with force control
    # 3. Dense sand with displacement control (for comparison)

    sim_loose = ForceControlledSimulator(
        grid_size=64, domain_size=1.0,
        kc=0.0, kphi=1000.0, n=1.0,  # Softer sand
        c=0.0, phi=28.0, K=0.030
    )

    sim_dense = ForceControlledSimulator(
        grid_size=64, domain_size=1.0,
        kc=0.0, kphi=3000.0, n=1.2,  # Stiffer sand
        c=0.0, phi=34.0, K=0.020
    )

    # Displacement control simulator (same dense sand, but fixed depth)
    sim_disp = ForceControlledSimulator(
        grid_size=64, domain_size=1.0,
        kc=0.0, kphi=3000.0, n=1.2,
        c=0.0, phi=34.0, K=0.020
    )
    
    # Initialize all with flat bed
    flat_height = 0.10
    sim_loose.height[:, :] = flat_height
    sim_dense.height[:, :] = flat_height
    sim_disp.height[:, :] = flat_height
    
    # Blade parameters
    blade_width = 0.12
    normal_force = 1000  # N
    tangential_force = 50  # N
    
    print(f"\nBlade: width={blade_width}m, F_n={normal_force}N, F_t={tangential_force}N")
    print()
    
    # Force-controlled trenching in loose sand
    print("LOOSE SAND (kphi=1000):")
    result_loose = sim_loose.push_with_force(
        x0=0.2, y0=0.5, x1=0.8, y1=0.5,
        normal_force_N=normal_force,
        tangential_force_N=tangential_force,
        blade_width=blade_width
    )
    print(f"  Pressure: {result_loose['normal_pressure_kPa']:.1f} kPa")
    print(f"  Sinkage:  {result_loose['sinkage']*1000:.1f} mm")
    
    # Force-controlled trenching in dense sand
    print("\nDENSE SAND - FORCE CONTROL (kphi=3000):")
    result_dense = sim_dense.push_with_force(
        x0=0.2, y0=0.5, x1=0.8, y1=0.5,
        normal_force_N=normal_force,
        tangential_force_N=tangential_force,
        blade_width=blade_width
    )
    print(f"  Pressure: {result_dense['normal_pressure_kPa']:.1f} kPa")
    print(f"  Sinkage:  {result_dense['sinkage']*1000:.1f} mm")

    # Displacement-controlled trenching in dense sand (fixed depth)
    fixed_depth = 0.015  # 15mm - arbitrary choice!
    print(f"\nDENSE SAND - DISPLACEMENT CONTROL:")
    print(f"  Fixed depth: {fixed_depth*1000:.1f} mm (arbitrary!)")
    sim_disp.push_with_blade(
        x0=0.2, y0=0.5, x1=0.8, y1=0.5,
        indent_depth=fixed_depth,
        blade_width=blade_width
    )
    # Compute required force for this displacement
    required_pressure = sim_disp.compute_pressure_from_sinkage(fixed_depth, blade_width)
    blade_area = 0.6 * blade_width
    required_force = required_pressure * 1000.0 * blade_area
    print(f"  Required force: {required_force:.0f} N (computed)")

    print(f"\n=== COMPARISON ===")
    print(f"FORCE CONTROL:  {normal_force}N → {result_dense['sinkage']*1000:.1f}mm sinkage")
    print(f"DISPLACEMENT:   {required_force:.0f}N → {fixed_depth*1000:.1f}mm sinkage (fixed)")
    print(f"\nKey insight: Same force → Different sinkage in different materials!")
    print(f"  Loose/Dense ratio: {result_loose['sinkage']/result_dense['sinkage']:.2f}x")
    
    # Stabilize all three
    sim_loose.stabilize(max_iterations=200, tolerance=1e-6, verbose=False)
    sim_dense.stabilize(max_iterations=200, tolerance=1e-6, verbose=False)
    sim_disp.stabilize(max_iterations=200, tolerance=1e-6, verbose=False)
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))

    # 3D views - now with 3 comparisons
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(sim_loose.X, sim_loose.Y, sim_loose.height, cmap=cm.terrain, alpha=0.9)
    ax1.set_title(f'Loose Sand\nForce: {normal_force}N → {result_loose["sinkage"]*1000:.1f}mm', fontweight='bold')
    ax1.set_zlim(0, flat_height*1.1)
    ax1.view_init(elev=25, azim=45)

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(sim_dense.X, sim_dense.Y, sim_dense.height, cmap=cm.terrain, alpha=0.9)
    ax2.set_title(f'Dense Sand (Force Control)\nForce: {normal_force}N → {result_dense["sinkage"]*1000:.1f}mm', fontweight='bold')
    ax2.set_zlim(0, flat_height*1.1)
    ax2.view_init(elev=25, azim=45)

    ax2b = fig.add_subplot(2, 3, 3, projection='3d')
    ax2b.plot_surface(sim_disp.X, sim_disp.Y, sim_disp.height, cmap=cm.terrain, alpha=0.9)
    ax2b.set_title(f'Dense Sand (Displacement)\nFixed: {fixed_depth*1000:.1f}mm → {required_force:.0f}N', fontweight='bold')
    ax2b.set_zlim(0, flat_height*1.1)
    ax2b.view_init(elev=25, azim=45)
    
    # Cross-sections
    center_i = sim_loose.grid_size // 2
    profile_loose = sim_loose.height[center_i, :]
    profile_dense = sim_dense.height[center_i, :]
    profile_disp = sim_disp.height[center_i, :]

    ax3 = fig.add_subplot(2, 3, 4)
    ax3.plot(sim_loose.x, profile_loose, 'b-', linewidth=2, label=f'Loose (force: {normal_force}N)')
    ax3.plot(sim_dense.x, profile_dense, 'r-', linewidth=2, label=f'Dense (force: {normal_force}N)')
    ax3.plot(sim_disp.x, profile_disp, 'g--', linewidth=2, label=f'Dense (disp: {fixed_depth*1000:.1f}mm)')
    ax3.axhline(y=flat_height, color='gray', linestyle=':', alpha=0.5, label='Original surface')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Cross-Section: Force vs Displacement Control', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Force-Sinkage curves for different materials
    ax4 = fig.add_subplot(2, 3, 5)
    forces_range = np.linspace(100, 3000, 20)
    sinkages_loose = []
    sinkages_dense = []
    
    for F in forces_range:
        p = (F / (0.6 * blade_width)) / 1000  # Approximate blade area
        z_loose = (p / 1000.0) ** (1.0 / 1.0)
        z_dense = (p / 3000.0) ** (1.0 / 1.2)
        sinkages_loose.append(z_loose * 1000)
        sinkages_dense.append(z_dense * 1000)
    
    ax4.plot(sinkages_loose, forces_range, 'b-', linewidth=2, label='Loose (kphi=1000)')
    ax4.plot(sinkages_dense, forces_range, 'r-', linewidth=2, label='Dense (kphi=3000)')
    ax4.axhline(y=normal_force, color='green', linestyle='--', alpha=0.7, label=f'Force control: {normal_force}N')
    ax4.plot([fixed_depth*1000], [required_force], 'go', markersize=10, label=f'Displacement: {fixed_depth*1000:.1f}mm')
    ax4.set_xlabel('Sinkage (mm)')
    ax4.set_ylabel('Normal Force (N)')
    ax4.set_title('Bekker Curves', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # Pressure distribution
    ax5 = fig.add_subplot(2, 3, 6)
    pressures_range = np.linspace(0, 50, 50)
    sinkages_p_loose = [(p / 1000.0) ** (1.0 / 1.0) * 1000 for p in pressures_range]
    sinkages_p_dense = [(p / 3000.0) ** (1.0 / 1.2) * 1000 for p in pressures_range]
    
    ax5.plot(pressures_range, sinkages_p_loose, 'b-', linewidth=2, label='Loose')
    ax5.plot(pressures_range, sinkages_p_dense, 'r-', linewidth=2, label='Dense')
    ax5.set_xlabel('Pressure (kPa)')
    ax5.set_ylabel('Sinkage (mm)')
    ax5.set_title('Pressure-Sinkage', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/trenching_dynamic.png', dpi=150)
    print("\nSaved: trenching_dynamic.png")
    plt.show()


def test_force_displacement_equivalence():
    """
    Test that force-controlled and displacement-controlled give consistent results.
    
    Workflow:
    1. Apply force F → compute sinkage z_force
    2. Apply displacement z_force → compute required force F_required  
    3. Check: F ≈ F_required (should be close!)
    """
    print("\n" + "=" * 60)
    print("Equivalence Test: Force ↔ Displacement")
    print("=" * 60)
    
    sim = ForceControlledSimulator(
        grid_size=64, domain_size=1.0,
        kc=0.0, kphi=1528.0, n=1.1
    )
    
    blade_width = 0.12
    applied_force = 1000  # N
    blade_area = 0.6 * blade_width  # Approximate
    
    # Force → Sinkage
    pressure_kPa = (applied_force / blade_area) / 1000.0
    sinkage_from_force = sim.compute_sinkage_from_pressure(pressure_kPa, blade_width)
    
    # Sinkage → Force (inverse)
    pressure_back = sim.compute_pressure_from_sinkage(sinkage_from_force, blade_width)
    force_back = pressure_back * 1000.0 * blade_area
    
    print(f"\nForward: F={applied_force}N → z={sinkage_from_force*1000:.2f}mm")
    print(f"Inverse: z={sinkage_from_force*1000:.2f}mm → F={force_back:.1f}N")
    print(f"Error: {abs(applied_force - force_back)/applied_force * 100:.2f}%")
    
    if abs(applied_force - force_back) / applied_force < 0.01:
        print("✓ Bekker equations are invertible (error < 1%)")
    else:
        print("✗ Large error - check implementation")


if __name__ == "__main__":
    compare_control_modes()
    test_force_displacement_equivalence()
