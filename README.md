# Centroidal Dynamics MPC for NAO Locomotion

## Setup

Create Conda environment, install Pinocchio and MeshCat:

```bash
conda create -n nao-mpc python=3.12
conda activate nao-mpc
conda install pinocchio meshcat-python -c conda-forge
```

## Run code

Run a single solve of MPC optimization problem with:

```bash
python run_opti.py
```

Run the MPC in closed loop with:

```bash
python run_mpc.py
```

## Optimization Problem

The optimization problem uses a centroidal dynamics model, with the following decision variables:
- States: Centroidal momentum `h_com`, Generalized coordinates `q`
- Inputs: Joint velocities `v_j`, Ground reaction forces `forces`

Importantly, 4 ground reaction force frames are defined *per foot* (at each corner). Furthermore, one frame is defined at the center of each foot, to formulate the velocity constraints during contact and swing.

## Solver

Within the run scripts choose:
- "fatrop": Linux and Mac (better performance)
- "ipopt": Windows

## P&S Task

1. Read the PDF `nao-mpc.pdf`.
2. Implement the TODOs in `dynamics.py` and `optimization.py`.
3. To help debug, run `debug_fatrop.py`, which displays the constraint Jacobian of the optimization. You can also set `debug=True` in the scripts to print the solutions for `h_com, q, v_j, forces, foot_vel`. 
4. Once the MPC works, tune the weights in `optimization.py` and play with the commands/parameters (nodes, dt, etc.).
