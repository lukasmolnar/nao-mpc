import time
import numpy as np
import pinocchio as pin

from helpers import *
from optimization import Optimization


# Parameters
nao = NAO()
gait_type = "walk"
gait_nodes = 30
nodes = 15
dt = 0.02  # seconds

# User command
base_vel_des = np.array([0.2, 0, 0, 0, 0, 0])  # linear + angular
step_height = 0.02

# Print debug info
debug = True


def main():
    # Initialze robot
    nao.set_gait_sequence(gait_type, gait_nodes, step_height, dt)
    pin.computeAllTerms(nao.model, nao.data, nao.q0, np.zeros(nao.nv))
    print("NAO model: ", nao.model)

    # Setup optimization
    opti = Optimization(nao, nodes, dt)
    opti.setup_problem()
    opti.set_base_target(base_vel_des)

    # Solve
    x_init = np.concatenate((np.zeros(6), nao.q0))
    opti.update_initial_state(x_init)
    opti.update_contact_schedule(shift_idx=0)
    opti.init_solver(solver="fatrop")
    opti.solve(retract_all=True)

    # Debug
    if debug:
        for i in range(nodes):
            h = opti.h_sol[i]
            q = opti.q_sol[i]
            v_j = opti.v_j_sol[i]
            forces = opti.forces_sol[i]

            # Compute base velocity
            v_b = opti.dyn.base_vel_dynamics()(h, q, v_j)
            v_b = np.array(v_b).flatten()
            v = np.concatenate((v_b, v_j))

            # Compute foot frame velocities
            left_foot = opti.foot_center_frames[0]
            right_foot = opti.foot_center_frames[1]
            left_vel = opti.dyn.get_frame_velocity(left_foot)(q, v)
            right_vel = opti.dyn.get_frame_velocity(right_foot)(q, v)

            # Print desired info
            print("Node: ", i)
            print("h: ", h)
            print("q: ", q)
            print("v: ", v)
            print("forces: ", forces)
            print("left foot vel: ", left_vel)
            print("right foot vel: ", right_vel)

    # Visualize
    nao.robot.initViewer()
    nao.robot.loadViewerModel("pinocchio")
    for _ in range(50):
        for q in opti.q_sol:
            nao.robot.display(q)
            time.sleep(dt)


if __name__ == "__main__":
    main()
