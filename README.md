# MPCC-CLF-CBF for UAV Path-Following 🚁

This repository implements a **Nonlinear Model Predictive Contour Controller (MPCC)** combined with **Control Lyapunov Functions (CLF)** and **High-Order Control Barrier Functions (HOCBF)** for stable and safe trajectory tracking in UAVs.

The code is based on **CasADi 3.5.5** and **ACADOS (latest version)**, and has been **validated experimentally on a DJI Matrice 100** platform.

🔬 The full paper can be found at: [https://ieeeaccess.ieee.org/](Paper Final version)


---

## 🧠 Project Summary

This work proposes a novel path-following strategy for Unmanned Aerial Vehicles (UAVs) by combining:

- **MPCC**: Penalizes contouring and lag errors relative to a reference path.
- **CLF (Control Lyapunov Function)**: Ensures asymptotic stability toward the trajectory.
- **HOCBF (High-Order Control Barrier Functions)**: Enforces obstacle avoidance using second-order Lie derivatives.

The controller operates on a **quaternion-based UAV model** to avoid issues with Euler angle singularities, and outputs thrust and angular rates.

---

## 📦 Dependencies

```bash

- Python 3.8+
- [`casadi==3.5.5`](https://web.casadi.org/)
- [`acados`](https://github.com/acados/acados)
- `numpy`, `matplotlib`, `scipy`
- ROS (for real-time execution and simulation)
- ROS messages: `geometry_msgs`, `nav_msgs`
```
---

## 📁 Repository Structure

```bash
├── P_UAV_simple.py
├── V1_MPCC_SimpleModel_Quat_CLF_CBF.py
├── V2_MPCC_SimpleModel_Quat_CLF_CBF.py
├── T_MPC_SimpleModel_*.py
├── Functions_SimpleModel.py
├── fancy_plots.py
├── c_generated_code/
├── acados_ocp_*.json
├── 1_pose.png, 2_error_pose.png, 3_Time.png

```

**Key files:**

- `P_UAV_simple.py`: Initializes the UAV state (position, orientation).
- `V1_MPCC_SimpleModel_Quat_CLF_CBF.py`: Main MPCC + CLF-CBF controller using quaternions.
- `V2_MPCC_SimpleModel_Quat_CLF_CBF.py`: Same as above but uses Euler angles instead.
- `T_MPC_SimpleModel_*.py`: Classical MPC variations without contouring cost.
- `Functions_SimpleModel.py`: Contains system dynamics, rotation utilities, error functions, and CLF/CBF definitions.
- `fancy_plots.py`: Custom plotting functions.
- `c_generated_code/`: ACADOS-generated C code.
- `*.json`: ACADOS OCP problem descriptions.

---

## 🚀 How to Run

### 1. Initialize UAV Position

```bash
python3 P_UAV_simple.py
```

This step sets the initial state of the UAV (needed before the controller starts).

---

### 2. Run the MPCC + CLF-CBF Controller (Quaternion Version)

```bash
python3 V1_MPCC_SimpleModel_Quat_CLF_CBF.py
```

If you prefer to work with Euler angles:

```bash
python3 V2_MPCC_SimpleModel_Quat_CLF_CBF.py
```

---

### During execution:
- You will be asked if you want to use an existing solver (`a`) or create a new one (`n`).
- The controller publishes reference trajectories and predicted states.
- Final results are stored in `.mat` files for analysis.

---

## 📊 Output

- Time series plots of state evolution, error metrics, and computation time.
- Predicted trajectories and control actions.
- Optionally saved `.mat` files for MATLAB post-processing.

---

## 🔍 Code Walkthrough

### `Functions_SimpleModel.py` – Core Utilities

This file defines the system’s continuous dynamics and supporting mathematical functions:

- `f(x, u)`: Defines the UAV dynamics using quaternions.
- `rotation_utils`: Includes conversion between quaternions and rotation matrices.
- `error_functions`: Calculates contouring and lag errors based on path projection.
- `clf_terms`: Defines the CLF function and its time derivative.
- `cbf_terms`: Defines obstacle constraints using second-order Lie derivatives.

---

### `V1_MPCC_SimpleModel_Quat_CLF_CBF.py` – Main Controller

This script builds and solves the NMPC problem:

- **States**: Position `p`, orientation quaternion `q`, linear velocity `v`, angular velocity `w`.
- **Inputs**: Thrust and angular rates.
- **Objective Function**:
  - Minimize:
    - Contour error
    - Lag error
    - Quaternion error (log map)
    - Input variation
  - Maximize forward progress (using arc-length parameterization).
- **Constraints**:
  - UAV dynamics via CasADi symbolic model.
  - CLF: Enforces convergence.
  - CBF: Ensures obstacle avoidance (even for moving obstacles).
- **Solver**: Uses ACADOS for real-time iteration and fast computation.
- **Prediction Horizon**: Tunable based on desired time responsiveness and accuracy.

---

### `P_UAV_simple.py` – Initial Conditions

Before the controller is launched, this script publishes the initial state of the UAV:

- Simulated takeoff or manual state setting.
- Essential for proper warm-starting of the OCP solver.

---

### `fancy_plots.py` – Visualizations

Creates plots for:

- Position tracking
- Orientation error
- Contouring vs lag errors
- Solver computation time per iteration

---

## 🤝 Credits

Developed by:

- Bryan S. Guevara  
- José Varela-Aldás  
- Viviana Moya  
- Manuel Cardona  
- Daniel C. Gandolfo  
- Juan M. Toibero

Affiliations:  
1. Instituto de Automática, Universidad Nacional de San Juan – CONICET, San Juan, Argentina  
2. Facultad de Ciencias Técnicas, Universidad Internacional del Ecuador, Quito, Ecuador  
3. Facultad de Ingeniería y Ciencias Aplicadas, Universidad de las Américas, Quito, Ecuador  
4. Research Department, Universidad Don Bosco, Soyapango 1774, El Salvador

---

## 📌 Acknowledgments

This project was funded by:

- Universidad Indoamérica 🇪🇨  
- Universidad Internacional del Ecuador 🇪🇨  
- INAUT - Universidad Nacional de San Juan / CONICET 🇦🇷  

Experimental validation was conducted on a **DJI Matrice 100** quadrotor platform.

---
```

Let me know if you want a version with GitHub badges, Docker support, or extended installation instructions.


