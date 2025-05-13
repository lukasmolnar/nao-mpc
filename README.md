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

Importantly, 4 ground reaction force frames are defined *per foot* (at each corner). Furthermore, one frame is defined at the center of each foot, to formulate the contact/swing constraints.

## Solver

Within the run scripts choose:
- "fatrop": Linux and Mac (better performance)
- "ipopt": Windows