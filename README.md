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
python cylinder_crosssection.py    # Validates algorithm (pile collapse)
python test_simple_trench.py        # Trench on flat bed
```

## Core Files

1. **`sand_simulator.py`** - Main simulator class
2. **`cylinder_crosssection.py`** - Validation: cylinder collapses to 29° angle of repose
3. **`test_simple_trench.py`** - Demo: trench cuts into flat bed, sides avalanche in
4. **`requirements.txt`** - Dependencies (numpy, matplotlib)
5. **`.gitignore`** - Git ignore file (excludes .venv, __pycache__, *.png)

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

## Reference

Kim, W., Pavlov, C., & Johnson, A. M. (2019). Developing a Simple Model for Sand-Tool Interaction and Autonomously Shaping Sand. *arXiv:1908.02745*.
