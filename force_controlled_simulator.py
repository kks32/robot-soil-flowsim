"""
Force-Controlled Sand Simulator using Bekker-Wong Terramechanics Model

Extends the slope flow avalanche method with pressure-sinkage relationships.
Now we can control blade forces (normal and tangential) instead of just displacement.

References:
- Bekker, M. G. (1969). Introduction to Terrain-Vehicle Systems
- Wong, J. Y. (2008). Theory of Ground Vehicles
- Janosi, Z. & Hanamoto, B. (1961). Analytical determination of drawbar pull
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sand_simulator import SandSimulator


class ForceControlledSimulator(SandSimulator):
    """
    Extends SandSimulator with Bekker-Wong terramechanics for force-based control.
    
    Key additions:
    - Bekker pressure-sinkage relationship (normal force → sinkage)
    - Janosi-Hanamoto shear model (tangential force → displacement)
    - Force feedback from displaced volume
    """
    
    def __init__(self, grid_size=64, domain_size=1.0, angle_of_repose=29.0,
                 kc=0.0, kphi=1528.0, n=1.1, 
                 c=0.0, phi=30.0, K=0.025):
        """
        Initialize force-controlled simulator with Bekker-Wong parameters.
        
        Bekker Pressure-Sinkage Parameters (for dry sand):
        --------------------------------------------------
        kc : float
            Cohesive modulus of soil deformation (kN/m^(n+1))
            Typical: 0-5 for dry sand (≈0 for cohesionless sand)
        kphi : float
            Frictional modulus of soil deformation (kN/m^(n+2))
            Typical: 1000-5000 for sand (we use 1528 kN/m^n from literature)
        n : float
            Exponent of deformation (dimensionless)
            Typical: 0.5-1.5 for sand (we use 1.1)
        
        Janosi-Hanamoto Shear Parameters (for dry sand):
        ------------------------------------------------
        c : float
            Cohesion (kPa)
            Dry sand: ≈0 (cohesionless)
        phi : float
            Internal friction angle (degrees)
            Typical: 28-34° for loose sand, 30-40° for dense sand
        K : float
            Shear deformation modulus (m)
            Typical: 0.01-0.05m for sand
            
        Note: angle_of_repose and phi are related but not identical:
        - angle_of_repose: slope stability (surface flow)
        - phi: internal friction (shear resistance in bulk)
        """
        super().__init__(grid_size, domain_size, angle_of_repose)
        
        # Bekker pressure-sinkage parameters
        self.kc = kc        # Cohesive modulus (kN/m^(n+1))
        self.kphi = kphi    # Frictional modulus (kN/m^n)
        self.n = n          # Exponent
        
        # Janosi-Hanamoto shear parameters
        self.c = c                              # Cohesion (kPa)
        self.phi_internal = np.deg2rad(phi)     # Internal friction angle (rad)
        self.K = K                              # Shear deformation modulus (m)
        
        print(f"Force-Controlled Simulator initialized:")
        print(f"  Bekker parameters: kc={kc}, kphi={kphi}, n={n}")
        print(f"  Shear parameters: c={c} kPa, phi={phi}°, K={K}m")
    
    def compute_sinkage_from_pressure(self, pressure_kPa, blade_width):
        """
        Compute sinkage (z) from normal pressure using Bekker equation.
        
        Bekker equation: p = (kc/b + kphi) * z^n
        
        Solving for z: z = [p / (kc/b + kphi)]^(1/n)
        
        Parameters:
        -----------
        pressure_kPa : float
            Normal pressure on blade (kPa)
        blade_width : float
            Blade width (m)
        
        Returns:
        --------
        sinkage : float
            Vertical sinkage depth (m)
        """
        # Bekker coefficient
        bekker_coeff = self.kc / blade_width + self.kphi
        
        # Solve for sinkage
        if bekker_coeff > 0 and pressure_kPa > 0:
            sinkage = (pressure_kPa / bekker_coeff) ** (1.0 / self.n)
        else:
            sinkage = 0.0
        
        return sinkage
    
    def compute_pressure_from_sinkage(self, sinkage, blade_width):
        """
        Compute pressure from sinkage (inverse of above).
        
        p = (kc/b + kphi) * z^n
        
        Parameters:
        -----------
        sinkage : float
            Vertical sinkage depth (m)
        blade_width : float
            Blade width (m)
        
        Returns:
        --------
        pressure : float
            Normal pressure (kPa)
        """
        bekker_coeff = self.kc / blade_width + self.kphi
        pressure = bekker_coeff * (sinkage ** self.n)
        return pressure
    
    def compute_shear_displacement(self, shear_stress_kPa, normal_pressure_kPa):
        """
        Compute horizontal displacement from shear stress using Janosi-Hanamoto.
        
        Janosi-Hanamoto equation:
        τ = (c + σ*tan(φ)) * (1 - exp(-j/K))
        
        Where:
        τ = shear stress
        σ = normal stress (pressure)
        c = cohesion
        φ = internal friction angle
        j = shear displacement
        K = shear deformation modulus
        
        Solving for j:
        j = -K * ln(1 - τ/τ_max)
        where τ_max = c + σ*tan(φ)
        
        Parameters:
        -----------
        shear_stress_kPa : float
            Tangential shear stress (kPa)
        normal_pressure_kPa : float
            Normal pressure (kPa)
        
        Returns:
        --------
        displacement : float
            Horizontal shear displacement (m)
        """
        # Maximum shear stress (Mohr-Coulomb)
        tau_max = self.c + normal_pressure_kPa * np.tan(self.phi_internal)
        
        # Avoid division by zero
        if tau_max < 1e-6:
            return 0.0
        
        # Ratio of applied to maximum shear
        ratio = min(shear_stress_kPa / tau_max, 0.99)  # Cap at 0.99 to avoid log(0)
        
        # Solve for displacement
        if ratio > 0:
            displacement = -self.K * np.log(1.0 - ratio)
        else:
            displacement = 0.0
        
        return displacement
    
    def push_with_force(self, x0, y0, x1, y1, 
                       normal_force_N, tangential_force_N, blade_width):
        """
        Push blade through sand using force control (not displacement control).
        
        This is the KEY METHOD that uses Bekker-Wong!
        
        Workflow:
        1. Compute blade contact area
        2. Calculate pressures from forces
        3. Use Bekker to get sinkage (vertical indent)
        4. Use Janosi-Hanamoto to get shear displacement (horizontal)
        5. Apply deformation to heightmap
        6. Stabilize with slope flow avalanche
        
        Parameters:
        -----------
        x0, y0 : float
            Blade start position (m)
        x1, y1 : float
            Blade end position (m)
        normal_force_N : float
            Normal force pressing blade down (N)
        tangential_force_N : float
            Tangential force pushing blade forward (N)
        blade_width : float
            Blade width perpendicular to motion (m)
        
        Returns:
        --------
        results : dict
            {
                'sinkage': vertical indent depth (m),
                'shear_displacement': horizontal displacement (m),
                'normal_pressure': pressure from normal force (kPa),
                'shear_stress': shear stress (kPa),
                'volume_displaced': volume of sand removed (m³),
                'reaction_force_estimate': estimated resistance force (N)
            }
        """
        # Compute blade contact area
        blade_length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        contact_area = blade_length * blade_width  # m²
        
        # Convert forces to pressures (N/m² = Pa, then to kPa)
        normal_pressure_kPa = (normal_force_N / contact_area) / 1000.0
        shear_stress_kPa = (tangential_force_N / contact_area) / 1000.0
        
        # Bekker: pressure → sinkage
        sinkage = self.compute_sinkage_from_pressure(normal_pressure_kPa, blade_width)
        
        # Janosi-Hanamoto: shear stress → displacement
        shear_displacement = self.compute_shear_displacement(shear_stress_kPa, 
                                                             normal_pressure_kPa)
        
        # Apply sinkage (create depression in heightmap)
        # Store initial volume for reaction force estimate
        volume_before = np.sum(self.height) * self.dx**2
        
        self.push_with_blade(x0, y0, x1, y1, 
                           indent_depth=sinkage, 
                           blade_width=blade_width)
        
        volume_after = np.sum(self.height) * self.dx**2
        volume_displaced = volume_before - volume_after
        
        # Estimate reaction force from pressure over displaced area
        # This is approximate - real resistance comes from material flow
        reaction_force_estimate = normal_pressure_kPa * 1000.0 * contact_area
        
        return {
            'sinkage': sinkage,
            'shear_displacement': shear_displacement,
            'normal_pressure_kPa': normal_pressure_kPa,
            'shear_stress_kPa': shear_stress_kPa,
            'volume_displaced': volume_displaced,
            'contact_area': contact_area,
            'reaction_force_N': reaction_force_estimate
        }


def demo_force_vs_displacement():
    """
    Demonstrate force-controlled vs displacement-controlled trenching.
    """
    print("=" * 70)
    print("FORCE-CONTROLLED TRENCHING DEMO")
    print("Bekker-Wong Terramechanics Model")
    print("=" * 70)
    
    # Create force-controlled simulator
    # Using typical values for dry sand from literature
    sim = ForceControlledSimulator(
        grid_size=64,
        domain_size=1.0,
        angle_of_repose=29.0,
        kc=0.0,           # Dry sand has ~zero cohesion
        kphi=1528.0,      # From literature for sand (kN/m^n)
        n=1.1,            # Typical for sand
        c=0.0,            # No cohesion
        phi=30.0,         # Internal friction angle (degrees)
        K=0.025           # Shear deformation modulus (m)
    )
    
    # Start with flat bed
    flat_height = 0.10
    sim.height[:, :] = flat_height
    
    print(f"\nStarting with flat bed at {flat_height}m height")
    print()
    
    # Test different normal forces
    forces_to_test = [100, 500, 1000, 2000]  # Newtons
    blade_width = 0.12  # m
    tangential_force = 50  # N (small, just to initiate motion)
    
    print("Testing different normal forces:")
    print(f"  Blade width: {blade_width}m")
    print(f"  Tangential force: {tangential_force}N")
    print()
    
    results_list = []
    
    for normal_force in forces_to_test:
        # Reset to flat bed
        sim.height[:, :] = flat_height
        
        # Apply force-controlled push
        result = sim.push_with_force(
            x0=0.2, y0=0.5,
            x1=0.8, y1=0.5,
            normal_force_N=normal_force,
            tangential_force_N=tangential_force,
            blade_width=blade_width
        )
        
        results_list.append(result)
        
        print(f"Normal Force: {normal_force}N")
        print(f"  → Pressure: {result['normal_pressure_kPa']:.2f} kPa")
        print(f"  → Sinkage: {result['sinkage']*1000:.2f} mm")
        print(f"  → Volume displaced: {result['volume_displaced']*1e6:.2f} cm³")
        print()
    
    # Visualize force-sinkage relationship
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Force vs Sinkage
    forces = [r['normal_pressure_kPa'] * r['contact_area'] * 1000 for r in results_list]
    sinkages = [r['sinkage'] * 1000 for r in results_list]
    
    ax1.plot(sinkages, forces, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sinkage (mm)', fontsize=12)
    ax1.set_ylabel('Normal Force (N)', fontsize=12)
    ax1.set_title('Bekker Pressure-Sinkage Relationship', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Pressure vs Sinkage
    pressures = [r['normal_pressure_kPa'] for r in results_list]
    
    ax2.plot(sinkages, pressures, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sinkage (mm)', fontsize=12)
    ax2.set_ylabel('Pressure (kPa)', fontsize=12)
    ax2.set_title('Pressure-Sinkage Curve', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/force_control_demo.png', dpi=150)
    print("✓ Saved: force_control_demo.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("✓ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_force_vs_displacement()
