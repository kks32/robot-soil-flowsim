### Differentiable Granular Simulator: Math and Methods

This document outlines the exact equations and numerical methods required to implement a differentiable granular simulator using the Shallow Water Equations (SWE) augmented with  rheology and Dynamic Resistive Force Theory (DRFT).

The goal is to model a height field  and depth-averaged velocity  that evolves under gravity, friction, and tool interaction forces.

---

### **1. Governing Equations (Continuous Form)**

We solve the **Conservation of Mass** and **Conservation of Momentum** on a 2D domain.

#### **1.1 Mass Conservation (Continuity)**

Describes the transport of sand volume.

* : Height of the granular pile [m].
* : Depth-averaged velocity vector [m/s].

#### **1.2 Momentum Conservation**

Describes the acceleration of sand due to forces.

* : Acceleration due to gravity [].
* : Friction source term (resists motion).
* : Momentum source term from the tool (injects motion).

---

### **2. Constitutive Laws (The "Sand" Physics)**

#### **2.1  Rheology (Friction)**

This law governs the transition from static solid to flowing fluid. It calculates an effective friction coefficient  based on the local strain rate.

**Step 1: Calculate Strain Rate ()**
Approximated from velocity gradients:

**Step 2: Calculate Inertial Number ()**
A dimensionless number comparing inertial forces to pressure forces.

* : Mean grain diameter [m].
* : Hydrostatic pressure estimate.

**Step 3: Calculate Effective Friction ()**
Using the Jop et al. (2006) formulation:

* : Static friction coefficient ().
* : Dynamic friction limit ().
* : Material constant.

**Step 4: Friction Source Term ()**
The force opposes the velocity direction.

---

#### **2.2 Dynamic Resistive Force Theory (Tool Interaction)**

This models the tool as a momentum source that "splashes" sand based on impact velocity.

**Force Magnitude ():**

* : Depth of tool in sand.
* : Velocity of the tool relative to the ground.
* : Static resistance coefficient.
* : Dynamic (inertial) coefficient. This  term is critical for "flinging."

**Source Term ():**
Applied only to cells  overlapping the tool's footprint.

* Units: Force per unit mass [].

---

### **3. Numerical Implementation (Discretization)**

We use a **Finite Volume Method (FVM)** or a simplified **Finite Difference** scheme on a staggered or collocated grid. For differentiability in PyTorch, a robust first-order **Upwind Scheme** or **Lax-Friedrichs** is preferred to avoid gradient explosions.

#### **3.1 Grid Definitions**

* : Height at cell  at time .
* : Velocities at cell .

#### **3.2 Discrete Gradients (Central Difference)**

#### **3.3 Time Integration (Semi-Implicit Euler)**

Update  first, then update  using the new  for stability.

**Update Height ():**
Using a conservative flux form:

Where fluxes  are computed using **Upwinding** (don't flux from empty cells):

**Update Momentum ():**

* **Advection :** Use semi-Lagrangian or upwind gradients.
* **Friction:** Treat implicitly or clamp to zero if velocity is very small (stop condition).
* **Stability:** Clamp . Set  where .

---

### **4. Differentiable Control Loop**

To train the controller, we unroll this simulator for  steps.

**Loss Function:**

**Gradients:**
We rely on Automatic Differentiation (AD) to compute:

Because the simulator is built from differentiable tensor operations (), this gradient flows naturally from the final shape error back to the tool velocity.

---

### **5. Python Implementation Structure**
import torch
import torch.nn.functional as F

class GranularSimulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.g = 9.81
        self.dx = 0.01  # 1cm grid
        self.dt = 0.001 # 1ms step
        self.rho = 1500.0 # kg/m3

        # Rheology Params
        self.d = 0.002 # grain diameter
        self.mu_s = 0.38 # tan(21)
        self.mu_2 = 0.64 # tan(33)
        self.I_0 = 0.279

        # DRFT Params
        self.C_drag = 5.0
        self.C_splash = 1.5

    def diff_x(self, f):
        # Central difference with padding
        f_pad = F.pad(f, (1,1,0,0), mode='replicate')
        return (f_pad[:,:,:,2:] - f_pad[:,:,:,:-2]) / (2*self.dx)

    def diff_y(self, f):
        f_pad = F.pad(f, (0,0,1,1), mode='replicate')
        return (f_pad[:,:,2:,:] - f_pad[:,:,:-2,:]) / (2*self.dx)

    def forward(self, h, u, v, tool_pos, tool_vel):
        """
        Inputs:
          h: (B, 1, N, N) Height
          u, v: (B, 1, N, N) Velocity components
          tool_pos: (B, 3) [x, y, z]
          tool_vel: (B, 2) [vx, vy]
        """
        
        # 1. Gradients
        dh_dx = self.diff_x(h)
        dh_dy = self.diff_y(h)
        du_dx = self.diff_x(u)
        du_dy = self.diff_y(u)
        dv_dx = self.diff_x(v)
        dv_dy = self.diff_y(v)

        # 2. Rheology (mu(I))
        # Strain rate magnitude
        strain = torch.sqrt(du_dx**2 + dv_dy**2 + 0.5*(du_dy + dv_dx)**2 + 1e-6)
        
        # Inertial Number
        pressure_depth = torch.clamp(h, min=1e-3)
        I = (strain * self.d) / torch.sqrt(self.g * pressure_depth)
        
        # Friction Coeff
        mu_eff = self.mu_s + (self.mu_2 - self.mu_s) / (self.I_0/(I+1e-6) + 1.0)
        
        # Friction Force (S_fric)
        vel_mag = torch.sqrt(u**2 + v**2 + 1e-6)
        # S_fric = -mu * g * h * (u/|u|)
        # Normalized to momentum source: -mu * g * (u/|u|)
        fric_x = -mu_eff * self.g * (u / vel_mag)
        fric_y = -mu_eff * self.g * (v / vel_mag)

        # 3. DRFT Tool Force
        # Create coordinate grid
        B, _, H, W = h.shape
        # ... (Grid generation code) ...
        
        # Mask tool
        # dist = sqrt((x-tx)^2 + (y-ty)^2)
        # mask = sigmoid(radius - dist)
        
        # Immersion
        # z_imm = relu(h - tool_z) * mask
        
        # Force Magnitude
        # F = z_imm * (C_drag + C_splash * |v_tool|^2)
        
        # Source Term (Force / (rho * h))
        # source_x = (F * tool_dir_x) / (rho * h)

        # 4. Updates
        # Height (Mass Cons)
        # dh/dt = - d(hu)/dx - d(hv)/dy
        flux_x = u*h # Simplified flux
        flux_y = v*h
        h_new = h - self.dt * (self.diff_x(flux_x) + self.diff_y(flux_y))
        
        # Velocity (Momentum Cons)
        # du/dt = -(u grad u) - g grad h + Fric + Tool
        advect_u = u*du_dx + v*du_dy
        u_new = u + self.dt * (-advect_u - self.g*dh_dx + fric_x + source_x)
        
        v_new = v + self.dt * (-advect_v - self.g*dh_dy + fric_y + source_y)

        return h_new, u_new, v_new
