# Sand Simulation - Slope Flow Avalanche Method

Implementation of the slope flow avalanche method from:
**"Developing a Simple Model for Sand-Tool Interaction and Autonomously Shaping Sand"**  
by Kim et al. (CMU, 2019)

## Quick Start

```bash
# Setup
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run demos
python cylinder_crosssection.py       # Validates algorithm (pile collapse)
python trench.py                       # Displacement-controlled trenching
python trenching_dynamic.py            # Force vs displacement comparison
```

## Core Files

1. **`sand_simulator.py`** - Base simulator (slope flow avalanche)
2. **`force_controlled_simulator.py`** - Bekker-Wong terramechanics extension
3. **`cylinder_crosssection.py`** - Validation: cylinder collapses to 29°
4. **`test_simple_trench.py`** - Demo: displacement-controlled trenching
5. **`requirements.txt`** - Dependencies (numpy, matplotlib)
6. **`.gitignore`** - Git ignore file (excludes .venv, __pycache__, *.png)

## Algorithm

### Soil Erosion (Equations 1-2 from paper)

```
Flow: q = k·Δx·(∂h/∂x - b_repose)   when ∂h/∂x > b_repose
Height update: Δh = (1/A) · Σ q
```

**Parameters:**
- `k = Δx²/8` (flow rate constant for 8-connectivity)
- `b_repose = tan(29°) ≈ 0.554` (angle of repose)
- Grid: 64×64 cells

### How It Works

1. **Height-map representation**: 2D grid where each cell stores sand height
2. **Flow check**: For each cell, check slope to all 8 neighbors
3. **Avalanche**: If slope > angle of repose, sand flows to neighbor
4. **Iterate**: Repeat until all slopes ≤ angle of repose (stable)

## Validation

**Cylinder Collapse Test:**
- Initial: Cylinder with vertical walls (infinite slope)
- Final: Converges to ~29.5° slope ✓
- Proof: Algorithm correctly enforces angle of repose

**Trench Test:**
- Start: Flat bed at 0.10m height
- Cut: Blade creates 0.05m deep trench (steep 72° walls)
- Stabilize: Sides avalanche into trench → 29° stable slopes ✓

## Force-Controlled Extension (Bekker-Wong)

The `ForceControlledSimulator` adds terramechanics models for realistic force-based control:

### Bekker Pressure-Sinkage
```
p = (kc/b + kphi) * z^n
```
- Given normal force F → compute pressure p → solve for sinkage z
- **Parameters**: kc=0, kphi=1528 kN/m^n, n=1.1 (dry sand)

### Janosi-Hanamoto Shear
```
τ = (c + σ*tan(φ)) * (1 - exp(-j/K))
```
- Given tangential force F_t → compute shear stress τ → solve for displacement j
- **Parameters**: c=0, φ=30°, K=0.025m (dry sand)

### Usage Example
```python
from force_controlled_simulator import ForceControlledSimulator

sim = ForceControlledSimulator()
result = sim.push_with_force(
    x0=0.2, y0=0.5, x1=0.8, y1=0.5,
    normal_force_N=1000,      # Push down with 1000N
    tangential_force_N=50,     # Push forward with 50N
    blade_width=0.12
)

print(f"Sinkage: {result['sinkage']*1000:.1f} mm")  # ~14mm for 1000N
```

## References

- Kim, W., Pavlov, C., & Johnson, A. M. (2019). Developing a Simple Model for Sand-Tool Interaction and Autonomously Shaping Sand. *arXiv:1908.02745*.
- Bekker, M. G. (1969). Introduction to Terrain-Vehicle Systems. University of Michigan Press.
- Wong, J. Y. (2008). Theory of Ground Vehicles. John Wiley & Sons.
