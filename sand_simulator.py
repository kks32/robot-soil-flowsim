"""
Sand Simulation using Slope Flow Avalanche Method
Based on "Developing a Simple Model for Sand-Tool Interaction" by Kim et al.
Implements height-map approach with angle of repose constraint.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class SandSimulator:
    """
    Simulates granular material (sand) using height-map approach.
    Implements soil erosion based on angle of repose constraint.
    """

    def __init__(self, grid_size=64, domain_size=1.0, angle_of_repose=29.0):
        """
        Initialize sand simulator.

        Parameters:
        -----------
        grid_size : int
            Number of grid cells in each dimension (grid_size x grid_size)
        domain_size : float
            Physical size of domain in meters
        angle_of_repose : float
            Angle of repose in degrees (default 29° from paper)
        """
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size  # Grid spacing

        # Angle of repose
        self.angle_of_repose = np.deg2rad(angle_of_repose)
        self.b_repose = np.tan(self.angle_of_repose)  # Slope threshold

        # Flow rate constant (Equation 4 from paper: k = Δx²/8 for 8-connectivity)
        self.k = (self.dx ** 2) / 8.0

        # Initialize height-map (flat surface)
        self.height = np.zeros((grid_size, grid_size))

        # Create coordinate grids for visualization
        self.x = np.linspace(0, domain_size, grid_size)
        self.y = np.linspace(0, domain_size, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Track tool occupancy so no sand flows into the tool footprint
        self.tool_mask = np.zeros((grid_size, grid_size), dtype=bool)

    def _valid_index(self, i, j):
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size

    def _clear_tool_mask(self):
        self.tool_mask.fill(False)

    def _advance_outside_tool(self, i, j, step_i, step_j):
        """Advance along a direction until leaving the current tool footprint."""
        ti, tj = i + step_i, j + step_j
        if step_i == 0 and step_j == 0:
            return i, j
        while self._valid_index(ti, tj) and self.tool_mask[ti, tj]:
            ti += step_i
            tj += step_j
        return ti, tj

    def _deposit_material(self, i, j, amount, forward_step, perp_step):
        """Distribute removed material in front of the blade."""
        if amount <= 0:
            return

        targets = []
        fi, fj = self._advance_outside_tool(i, j, forward_step[0], forward_step[1])
        targets.append((fi, fj, 0.6))

        li, lj = self._advance_outside_tool(fi, fj, perp_step[0], perp_step[1])
        targets.append((li, lj, 0.2))

        ri, rj = self._advance_outside_tool(fi, fj, -perp_step[0], -perp_step[1])
        targets.append((ri, rj, 0.2))

        remaining = amount
        for ti, tj, fraction in targets:
            if fraction <= 0 or not self._valid_index(ti, tj):
                continue
            if self.tool_mask[ti, tj]:
                continue
            delta = amount * fraction
            self.height[ti, tj] += delta
            remaining -= delta

        if remaining > 1e-12:
            # Fallback: force push the mass to the front cell
            self.height[fi, fj] += remaining

    def _apply_tool_displacement(self, footprint, forward_step, target_height):
        """Remove material inside the footprint and push it forward."""
        move_i, move_j = forward_step
        if move_i == 0 and move_j == 0:
            move_j = 1  # default forward direction
        perp_i, perp_j = -move_j, move_i

        for (ci, cj) in footprint:
            self.tool_mask[ci, cj] = True
            current_height = self.height[ci, cj]
            if current_height <= target_height:
                continue
            excess = current_height - target_height
            self.height[ci, cj] = target_height
            self._deposit_material(ci, cj, excess, (move_i, move_j), (perp_i, perp_j))

    def compute_flow(self):
        """
        Vectorized compute_flow using numpy slicing (interior cells only).
        Returns delta_h same shape as self.height.
        """
        h = self.height
        N = self.grid_size

        # output
        delta_h = np.zeros_like(h)

        # quick aliases
        b = self.b_repose
        coeff = (self.k / self.dx)  # same as before

        # operate on interior region only (indices 1 .. N-2 inclusive)
        # center slice size: (N-2, N-2)
        c = h[1:-1, 1:-1]
        tool_c = self.tool_mask[1:-1, 1:-1]

        # neighbor slices aligned with center c (all shape (N-2, N-2))
        N_slice  = h[0:-2, 1:-1]   # North
        NE_slice = h[0:-2, 2:  ]   # North-East
        E_slice  = h[1:-1, 2:  ]   # East
        SE_slice = h[2:  , 2:  ]   # South-East
        S_slice  = h[2:  , 1:-1]   # South
        SW_slice = h[2:  , 0:-2]   # South-West
        W_slice  = h[1:-1, 0:-2]   # West
        NW_slice = h[0:-2, 0:-2]   # North-West

        # corresponding tool masks for neighbor cells
        tool_N  = self.tool_mask[0:-2, 1:-1]
        tool_NE = self.tool_mask[0:-2, 2:  ]
        tool_E  = self.tool_mask[1:-1, 2:  ]
        tool_SE = self.tool_mask[2:  , 2:  ]
        tool_S  = self.tool_mask[2:  , 1:-1]
        tool_SW = self.tool_mask[2:  , 0:-2]
        tool_W  = self.tool_mask[1:-1, 0:-2]
        tool_NW = self.tool_mask[0:-2, 0:-2]

        d_diag = self.dx * np.sqrt(2.0)
        d = [self.dx, d_diag, self.dx, d_diag, self.dx, d_diag, self.dx, d_diag]

        neighs = [
            (N_slice,  tool_N,  d[0], (slice(0, N-2), slice(1, N-1))),  
            (NE_slice, tool_NE, d[1], (slice(0, N-2), slice(2, N  ))),   
            (E_slice,  tool_E,  d[2], (slice(1, N-1), slice(2, N  ))),   
            (SE_slice, tool_SE, d[3], (slice(2, N  ), slice(2, N  ))),   
            (S_slice,  tool_S,  d[4], (slice(2, N  ), slice(1, N-1))),  
            (SW_slice, tool_SW, d[5], (slice(2, N  ), slice(0, N-2))),  
            (W_slice,  tool_W,  d[6], (slice(1, N-1), slice(0, N-2))),  
            (NW_slice, tool_NW, d[7], (slice(0, N-2), slice(0, N-2)))  
        ]

        # center destination slice in delta_h
        center_dest = (slice(1, N-1), slice(1, N-1))

        # compute flows for each neighbor and accumulate
        for neigh_arr, tool_neigh, dist, neigh_dest in neighs:
            slope = (c - neigh_arr) / dist  # positive if center higher

            # valid locations: slope > b_repose and neither center nor neighbor are tool-occupied
            valid = (slope > b) & (~tool_c) & (~tool_neigh)

            if not np.any(valid):
                continue

            # compute q only where valid
            q = np.zeros_like(slope)
            q[valid] = coeff * (slope[valid] - b)

            # accumulate: center loses q, neighbor gains q
            delta_h[center_dest] -= q
            delta_h[neigh_dest] += q

        return delta_h

    def update_step(self):
        """
        Perform one update step of soil erosion.

        Returns:
        --------
        max_change : float
            Maximum height change in this step (for convergence check)
        """
        delta_h = self.compute_flow()
        self.height += delta_h
        return np.max(np.abs(delta_h))

    def stabilize(self, max_iterations=500, tolerance=1e-6, verbose=True):
        """
        Iteratively update until stable (slope everywhere ≤ angle of repose).

        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance (max height change)
        verbose : bool
            Print convergence information

        Returns:
        --------
        n_iterations : int
            Number of iterations until convergence
        """
        for iteration in range(max_iterations):
            max_change = self.update_step()

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max change = {max_change:.6f}")

            if max_change < tolerance:
                if verbose:
                    print(f"Converged in {iteration} iterations")
                return iteration

        if verbose:
            print(f"Warning: Did not converge after {max_iterations} iterations")
        return max_iterations

    def add_pile(self, center_x, center_y, height, radius):
        """
        Add a conical pile of sand at specified location.

        Parameters:
        -----------
        center_x, center_y : float
            Center position in physical coordinates [0, domain_size]
        height : float
            Height of pile
        radius : float
            Radius of pile base
        """
        # Convert to grid indices
        cx = int(center_x / self.domain_size * self.grid_size)
        cy = int(center_y / self.domain_size * self.grid_size)
        r_grid = int(radius / self.domain_size * self.grid_size)

        for i in range(max(0, cx - r_grid), min(self.grid_size, cx + r_grid)):
            for j in range(max(0, cy - r_grid), min(self.grid_size, cy + r_grid)):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist < r_grid:
                    # Conical pile
                    h_add = height * (1 - dist / r_grid)
                    self.height[i, j] += h_add

    def push_with_blade(
        self,
        x0,
        y0,
        x1,
        y1,
        indent_depth=None,
        blade_width=0.1,
        *,
        blade_depth=None,
        surface_height=None,
        relax_iterations=3,
        record_history=False,
        conserve_mass=True,
    ):
        """
        Create an indentation (trench) with a rectangular blade from (x0, y0) to (x1, y1).
        The blade simply creates a depression in the heightmap.

        Parameters:
        -----------
        x0, y0 : float
            Start position in physical coordinates
        x1, y1 : float
            End position in physical coordinates
        indent_depth : float
            Depth of indentation (how much to lower the surface)
        blade_width : float
            Width of blade
        """
        # Support legacy keyword `blade_depth`
        if indent_depth is None:
            if blade_depth is None:
                raise ValueError("indent_depth or blade_depth must be specified")
            indent_depth = blade_depth

        if surface_height is None:
            surface_height = float(np.max(self.height))

        target_height = max(surface_height - indent_depth, 0.0) if conserve_mass else None

        history = []

        def capture(step_idx):
            if record_history:
                history.append(
                    {
                        "step": step_idx,
                        "height": self.height.copy(),
                        "tool_mask": self.tool_mask.copy(),
                    }
                )

        step_counter = 0
        capture(step_counter)

        # Convert to grid coordinates
        i0 = int(y0 / self.domain_size * self.grid_size)
        j0 = int(x0 / self.domain_size * self.grid_size)
        i1 = int(y1 / self.domain_size * self.grid_size)
        j1 = int(x1 / self.domain_size * self.grid_size)

        # Number of steps along the path
        n_segments = max(abs(i1 - i0), abs(j1 - j0), 1)

        # Interpolate path (include both endpoints)
        i_path = np.linspace(i0, i1, n_segments + 1).astype(int)
        j_path = np.linspace(j0, j1, n_segments + 1).astype(int)

        # Blade width in grid cells
        w_grid = max(1, int(blade_width / self.domain_size * self.grid_size))

        carved_rows = set()

        # Create indentation along path
        for step in range(n_segments + 1):
            i_current = i_path[step]
            j_current = j_path[step]

            if step < n_segments:
                dir_i = i_path[step + 1] - i_current
                dir_j = j_path[step + 1] - j_current
            elif step > 0:
                dir_i = i_current - i_path[step - 1]
                dir_j = j_current - j_path[step - 1]
            else:
                dir_i, dir_j = 0, 1

            move_i = int(np.sign(dir_i))
            move_j = int(np.sign(dir_j))
            if move_i == 0 and move_j == 0:
                move_j = 1

            perp_i = -move_j
            perp_j = move_i

            footprint = []
            for w in range(-w_grid // 2, w_grid // 2 + 1):
                i_blade = i_current + w * perp_i
                j_blade = j_current + w * perp_j

                if not self._valid_index(i_blade, j_blade):
                    continue
                footprint.append((i_blade, j_blade))

            self._clear_tool_mask()
            if conserve_mass:
                self._apply_tool_displacement(footprint, (move_i, move_j), target_height)
            else:
                for (ci, cj) in footprint:
                    if not self._valid_index(ci, cj):
                        continue
                    self.tool_mask[ci, cj] = True
                    self.height[ci, cj] = max(self.height[ci, cj] - indent_depth, 0.0)
                    carved_rows.add(ci)
            step_counter += 1
            capture(step_counter)

            for _ in range(max(relax_iterations, 0)):
                self.update_step()
            step_counter += 1
            capture(step_counter)

        self._clear_tool_mask()

        if record_history:
            step_counter += 1
            capture(step_counter)

        if not conserve_mass and carved_rows:
            target_height = max(surface_height - indent_depth, 0.0)
            epsilon = 1e-9
            for ri in carved_rows:
                row = self.height[ri]
                carved_cols = np.where(row < surface_height - epsilon)[0]
                if carved_cols.size == 0:
                    continue
                splits = np.where(np.diff(carved_cols) > 1)[0] + 1
                segments = np.split(carved_cols, splits)
                for segment in segments:
                    start, end = segment[0], segment[-1]
                    row[start:end + 1] = np.minimum(row[start:end + 1], target_height)

        if record_history:
            return history
        return None

    def visualize_3d(self, title="Sand Height Map", show=True):
        """
        Create 3D visualization of current height-map.

        Parameters:
        -----------
        title : str
            Plot title
        show : bool
            Whether to display the plot

        Returns:
        --------
        fig, ax : matplotlib figure and axis
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        surf = ax.plot_surface(self.X, self.Y, self.height,
                               cmap=cm.terrain,
                               linewidth=0,
                               antialiased=True,
                               alpha=0.9)

        # Add contour lines on the bottom
        ax.contour(self.X, self.Y, self.height,
                   zdir='z',
                   offset=np.min(self.height) - 0.02,
                   cmap=cm.terrain,
                   alpha=0.5)

        # Labels and title
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Height (m)', fontsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Height (m)')

        # Set viewing angle
        ax.view_init(elev=30, azim=45)

        if show:
            plt.show()

        return fig, ax


def main():
    """
    Main demonstration: Create sand pile and interact with rectangular blade.
    """
    print("=" * 70)
    print("Sand Simulation using Slope Flow Avalanche Method")
    print("Based on Kim et al. (2019)")
    print("=" * 70)

    # Create simulator
    print("\n1. Initializing simulator...")
    sim = SandSimulator(grid_size=64, domain_size=1.0, angle_of_repose=29.0)

    # Add initial sand pile
    print("2. Adding initial sand pile...")
    sim.add_pile(center_x=0.5, center_y=0.5, height=0.15, radius=0.2)

    # Visualize initial state
    print("3. Stabilizing initial pile...")
    sim.visualize_3d(title="Initial Sand Pile (Before Stabilization)", show=False)

    # Stabilize
    iterations = sim.stabilize(max_iterations=500, tolerance=1e-6, verbose=True)

    # Visualize after stabilization
    print("4. Visualizing stabilized pile...")
    sim.visualize_3d(title=f"Stabilized Sand Pile (After {iterations} iterations)", show=False)

    # Interact with blade
    print("\n5. Pushing sand with rectangular blade...")
    print("   Blade path: (0.2, 0.3) → (0.7, 0.6)")
    sim.push_with_blade(x0=0.2, y0=0.3, x1=0.7, y1=0.6,
                        blade_depth=0.05, blade_width=0.1)

    # Visualize after blade interaction
    sim.visualize_3d(title="After Blade Push (Before Stabilization)", show=False)

    # Stabilize again
    print("\n6. Stabilizing after blade interaction...")
    iterations = sim.stabilize(max_iterations=500, tolerance=1e-6, verbose=True)

    # Final visualization
    print("7. Final visualization...")
    sim.visualize_3d(title=f"Final Result (After {iterations} iterations)", show=False)

    # Add another blade pass
    print("\n8. Second blade pass...")
    print("   Blade path: (0.6, 0.2) → (0.4, 0.8)")
    sim.push_with_blade(x0=0.6, y0=0.2, x1=0.4, y1=0.8,
                        blade_depth=0.04, blade_width=0.08)

    sim.visualize_3d(title="After Second Blade Push (Before Stabilization)", show=False)

    print("\n9. Final stabilization...")
    iterations = sim.stabilize(max_iterations=500, tolerance=1e-6, verbose=True)

    sim.visualize_3d(title=f"Final Stabilized Result (After {iterations} iterations)", show=True)

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
