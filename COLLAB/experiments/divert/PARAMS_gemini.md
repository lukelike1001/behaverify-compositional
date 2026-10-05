# Independent Kinematic Parameter Derivation (`PARAMS_gemini.md`)

**Author:** Gemini  
**Basis:** Independent kinematic re-derivation from Section 6.2 & 6.3 of `COLLAB/CORA_ATTACK.md`, grounded in Hobbs et al. (*IEEE Control Systems Magazine* 2023) viability kernel and Pontryagin's maximum principle. Written prior to inspecting Claude's `plant.py`.

---

### 1. Plant Dynamics & Physical Constants
- **State space:** 4-D planar double integrator: $(x, y, v_x, v_y) \in \mathbb{R}^4$
- **Control input:** Planar acceleration $(a_x, a_y) \in [-a_{max}, a_{max}]^2$
- **Time step:** $dt = 0.1\text{ s}$
- **Aerodynamic drag / damping:** $c_d = 0.1\text{ s}^{-1}$
- **Integration equations:**
  $$x' = x + v_x \cdot dt, \quad y' = y + v_y \cdot dt$$
  $$v_x' = v_x + (a_x - c_d v_x) \cdot dt, \quad v_y' = v_y + (a_y - c_d v_y) \cdot dt$$

---

### 2. Control Authority & Speed Limits (The 40-Tick Wall Bound)
- **Speed Limit ($s_{max}$):** $1.0\text{ m/s}$
- **Maximum Acceleration ($a_{max}$):** $2.0\text{ m/s}^2$ ($\approx 0.20g$)
  - *Derivation & Rationale:* At $a_{max} = 1.0\text{ m/s}^2$, Pontryagin time-optimal return across distance $D \approx 1.25\text{ m}$ plus braking requires $\approx 37$ to $39$ ticks of recovery alone, bursting the compositional 40-tick wall ($246$ s). At $a_{max} = 2.0\text{ m/s}^2$:
    - Braking from $s_{max} = 1.0\text{ m/s}$ to rest takes $M = 5\text{ ticks}$ ($0.5$ s).
    - Divert return across $D \approx 1.04\text{ m}$ takes $1.25 \times 2\sqrt{D/a_{max}} \approx 18\text{ ticks}$ ($1.8$ s).
    - Recovery total $N = M + t_{return} = 23\text{ ticks}$.
    - Cruise to hazard: $8\text{ ticks}$.
    - Total mission: $31\text{ ticks}$, comfortably below the 40-tick wall with $9$ ticks of margin.

---

### 3. Viability Kernel & Buffer Sizing ($B^*$)
- **Continuous stopping distance from $s_{max} = 1.0\text{ m/s}$ under $a_{brake} = -2.0\text{ m/s}^2$ and $c_d = 0.1\text{ s}^{-1}$:**
  $$d_{stop}(v_n) = \frac{v_n}{c_d} - \frac{a_{max}}{c_d^2} \ln\left(1 + \frac{c_d v_n}{a_{max}}\right) = 10 - 200 \ln(1.05) \approx 0.2419\text{ m}$$
- **Discrete detection latency ($1$ tick at $v_{entry} = 1.0\text{ m/s}$):**
  $$d_{latency} = v_{entry} \cdot dt = 0.1000\text{ m}$$
- **Theoretical viability buffer:**
  $$B_{viability} = d_{stop} + d_{latency} = 0.3419\text{ m}$$
- **Certified buffer $B^*$ (with $\epsilon = 0.0581\text{ m}$ engineering margin):**
  $$B^* = 0.4000\text{ m}$$

---

### 4. Spatial Geometry & Regions
- **Start Region ($G_{start}$):**
  $$X_0 = [-0.10, 0.10] \times [-0.10, 0.10]\text{ m}$$
  $$V_0 = [-0.10, 0.10] \times [-0.10, 0.10]\text{ m/s}$$
- **Obstacle $\mathcal{O}$:**
  $$X_{\mathcal{O}} = [1.20, 1.80]\text{ m}, \quad Y_{\mathcal{O}} = [-0.50, 0.50]\text{ m}$$
  (Front face at $x_{obs\_front} = 1.20\text{ m}$; width $0.60\text{ m}$, span $1.00\text{ m}$).
- **Hazard Fence Boundary ($x_b$):**
  $$x_b = x_{obs\_front} - B^* = 1.20 - 0.40 = 0.80\text{ m} \quad (\text{for } y \in [-0.70, 0.70])$$
- **Target Clearance Margin ($\delta_{safe}$):**
  $$\delta_{safe} = 0.05\text{ m}$$
  - Worst-case physical overshoot when breaching fence at $x_b = 0.80$ with $v = 1.0\text{ m/s}$:
    $$x_{max} = 0.80 + 0.3419 = 1.1419\text{ m} < 1.20\text{ m}$$
    $$\text{Actual Minimum Clearance} = 1.20 - 1.1419 = 0.0581\text{ m} > \delta_{safe} = 0.0500\text{ m}$$
- **Nominal Goal ($G$):**
  $$X_G = [2.00, 2.40]\text{ m}, \quad Y_G = [0.80, 1.20]\text{ m}$$
  (Located beyond obstacle, requiring nominal cruise to steer around $\mathcal{O}$).
- **Arena Boundaries:**
  $$X_{arena} = [-0.50, 2.50]\text{ m}, \quad Y_{arena} = [-1.20, 1.20]\text{ m}$$
  - Floor / Ceiling: $y = \pm 1.20\text{ m}$
  - Out-of-bounds trigger: $|x| \ge 2.40\text{ m}$ or $|y| \ge 1.10\text{ m}$.

---

### 5. Discrete Counters & Node Thresholds
- **Stopping threshold:** $v_x^2 + v_y^2 \le v_{stop}^2 = 0.05^2 = 0.0025\text{ (m/s)}^2$
- **Docking threshold at $G_{start}$:** $(x, y) \in G_{start} \wedge (v_x^2 + v_y^2 \le v_{dock}^2 = 0.10^2 = 0.0100\text{ (m/s)}^2)$
- **Stuck threshold:** $k_{stuck} = 25\text{ ticks}$ ($2.5$ s)
- **Braking duration bound ($N_4$):** $M = 5\text{ ticks}$
- **End-to-end recovery bound ($N_1$):** $N = 23\text{ ticks}$
- **Mission horizon:** $K_{mission} = 31\text{ ticks}$ (strictly inside the 40-tick wall)
