import time
import numpy as np
import casadi as ca

from dynamics import Dynamics


class Optimization:
    def __init__(self, robot, nodes, dt):
        self.opti = ca.Opti()
        self.robot = robot
        self.nodes = nodes
        self.dt = dt

        self.model = robot.model
        self.data = robot.data
        self.gait_sequence = robot.gait_sequence
        self.foot_force_frames = robot.foot_force_frames
        self.foot_center_frames = robot.foot_center_frames
        self.n_contacts = robot.gait_sequence.n_contacts

        self.nq = robot.nq
        self.nv = robot.nv
        self.nj = robot.nj
        self.nf = robot.nf

        self.mass = self.data.mass[0]
        self.dyn = Dynamics(self.model, self.mass, self.foot_force_frames)

        # Nominal state
        self.x_nom = np.concatenate(([0] * 6, self.robot.q0))  # CoM + q0

        # Store solutions
        self.h_sol = []
        self.q_sol = []
        self.v_j_sol = []
        self.forces_sol = []

    def setup_problem(self):
        self.setup_variables()
        self.setup_parameters()
        self.setup_targets()
        self.setup_weights()
        self.setup_constraints()
        obj = self.setup_objective()
        self.opti.minimize(obj)

    def setup_variables(self):
        # States and Inputs
        self.nx = 6 + self.nq  # CoM + positions
        self.ndx_opt = 6 + self.nv  # CoM + position deltas
        self.nu_opt = self.nj + self.nf  # joint velocities + foot forces

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for _ in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

    def setup_parameters(self):
        self.x_init = self.opti.parameter(self.nx)  # initial state
        self.contact_schedule = self.opti.parameter(2, self.nodes) # in_contact: 0 or 1
        self.bezier_schedule = self.opti.parameter(2, self.nodes) # swing_phase: from 0 to 1
        self.base_vel_des = self.opti.parameter(6)  # target velocity (linear + angular)

    def setup_targets(self):
        # Desired state
        x_des = ca.vertcat(self.base_vel_des, self.robot.q0)  # CoM target + q0
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)

        # Desired input
        f_gravity = 9.81 * self.mass  # gravity force
        self.f_des = ca.repmat(ca.vertcat(0, 0, f_gravity / self.n_contacts), len(self.foot_force_frames), 1)
        self.u_des = ca.vertcat([0] * self.nj, self.f_des)  # zero velocities

    def setup_weights(self):
        # State weights
        Q_diag = np.concatenate((
            [1000] * 6,     # CoM
            [0] * 2,        # base x/y
            [500],          # base z
            [500] * 2,      # base rot x/y
            [0],            # base rot z
            [10] * self.nj  # joint positions
        ))

        # Input weights
        R_diag = np.concatenate((
            [1] * self.nj,    # joint velocities
            [1e-1] * self.nf  # foot forces
        ))

        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)

    def setup_objective(self):
        # Quadratic cost function
        obj = 0
        for i in range(self.nodes):
            # Track desired state and input
            dx = self.DX_opt[i]
            u = self.U_opt[i]
            err_dx = dx - self.dx_des
            err_u = u - self.u_des
            obj += err_dx.T @ self.Q @ err_dx
            obj += err_u.T @ self.R @ err_u

        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - self.dx_des
        obj += err_dx.T @ self.Q @ err_dx

        return obj

    def setup_constraints(self, mu=0.7):
        # Initial state
        self.opti.subject_to(self.DX_opt[0] == [0] * self.ndx_opt)

        for i in range(self.nodes):
            # Gather all state and input info
            dx = self.DX_opt[i]
            dh = dx[:6]  # delta h
            dq = dx[6:]  # delta q, not v
            q = self.get_q(i)
            v = self.get_v(i)
            forces = self.get_forces(i)

            # Dynamics constraint
            dx_next = self.DX_opt[i+1]
            dh_next = dx_next[:6]
            dq_next = dx_next[6:]
            h_dot = self.dyn.com_dynamics()(q, forces)
            self.opti.subject_to(dh_next == dh + h_dot * self.dt)
            self.opti.subject_to(dq_next == dq + v * self.dt)

            # Contact constraints (foot force frames)
            for idx, frame in enumerate(self.foot_force_frames):
                f_foot = forces[idx * 3 : (idx + 1) * 3]

                # Get contact info
                contact_idx = ca.if_else(idx < 4, 0, 1)  # first 4 contacts: left foot, then right foot
                in_contact = self.contact_schedule[contact_idx, i]

                # Friction cone
                f_normal = f_foot[2]
                f_tangent_square = f_foot[0]**2 + f_foot[1]**2
                self.opti.subject_to(in_contact * f_normal >= 0)
                self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)

                # Zero end-effector force
                self.opti.subject_to((1 - in_contact) * f_foot == [0] * 3)

            # Swing constraints (foot center frames)
            for idx, frame in enumerate(self.foot_center_frames):
                # Get contact and swing info
                in_contact = self.contact_schedule[idx, i]
                bezier_phase = self.bezier_schedule[idx, i]

                # Get velocity (linear + angular)
                vel = self.dyn.get_frame_velocity(frame)(q, v)
                vel_lin = vel[:3]
                vel_ang = vel[3:6]

                # Linear velocity x/y: Zero in contact, unconstrained in swing
                vel_lin_xy = vel_lin[:2]
                self.opti.subject_to(in_contact * vel_lin_xy == [0] * 2)

                # Linear velocity z: Zero in contact, track bezier in swing
                vel_lin_z = vel_lin[2]
                vel_bezier = self.gait_sequence.get_bezier_vel(bezier_phase)
                vel_diff = vel_lin_z - vel_bezier
                self.opti.subject_to(in_contact * vel_lin_z + (1 - in_contact) * vel_diff == 0)

                # Angular velocity x/y: Zero in contact AND in swing
                vel_ang_xy = vel_ang[:2]
                self.opti.subject_to(vel_ang_xy == [0] * 2)

                # Angular velocity z: Zero in contact, unconstrained in swing
                vel_ang_z = vel_ang[2]
                self.opti.subject_to(in_contact * vel_ang_z == 0)

            # Minimum distance between feet
            min_dist = 0.03
            l_heel_in = self.dyn.get_frame_position(self.foot_force_frames[0])(q)
            l_toe_in = self.dyn.get_frame_position(self.foot_force_frames[2])(q)
            r_heel_in = self.dyn.get_frame_position(self.foot_force_frames[4])(q)
            r_toe_in = self.dyn.get_frame_position(self.foot_force_frames[6])(q)
            heel_dist = l_heel_in - r_heel_in
            toe_dist = l_toe_in - r_toe_in
            heel_dist_xy = ca.sqrt(ca.sumsqr(heel_dist[:2]))
            toe_dist_xy = ca.sqrt(ca.sumsqr(toe_dist[:2]))
            self.opti.subject_to(heel_dist_xy >= min_dist)
            self.opti.subject_to(toe_dist_xy >= min_dist)

            # Warm start
            self.opti.set_value(self.n_contacts, self.gait_sequence.n_contacts)
            self.opti.set_initial(self.DX_opt[i], np.zeros(self.ndx_opt))
            self.opti.set_initial(self.U_opt[i], self.opti.value(self.u_des))

        # Warm start
        self.opti.set_initial(self.DX_opt[self.nodes], np.zeros(self.ndx_opt))

        # Store previous solution
        self.DX_prev = None
        self.U_prev = None

    def get_h(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[:6]

    def get_q(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[6:]

    def get_v(self, i):
        v_j = self.U_opt[i][:self.nj]  # joint velocities
        # Compute base velocity from dynamics
        h = self.get_h(i)
        q = self.get_q(i)
        v_b = self.dyn.base_vel_dynamics()(h, q, v_j)
        v = ca.vertcat(v_b, v_j)
        return v

    def get_forces(self, i):
        u = self.U_opt[i]
        return u[self.nj:]

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_contact_schedule(self, shift_idx=0):
        contact_schedule = self.gait_sequence.shift_contact_schedule(shift_idx)
        bezier_schedule = self.gait_sequence.shift_bezier_schedule(shift_idx)
        self.opti.set_value(self.contact_schedule, contact_schedule[:, :self.nodes])
        self.opti.set_value(self.bezier_schedule, bezier_schedule[:, :self.nodes])
        self.opti.set_value(self.n_contacts, self.gait_sequence.n_contacts)

    def set_base_target(self, base_vel_des):
        self.opti.set_value(self.base_vel_des, base_vel_des)

    def warm_start(self):
        # Shift previous solution
        # NOTE: No warm-start for last node, copying the 2nd last node performs worse.
        if self.DX_prev is not None:
            DX_init = self.DX_prev[1]
            for i in range(self.nodes):
                DX_diff = self.DX_prev[i+1] - DX_init
                self.opti.set_initial(self.DX_opt[i], DX_diff)

        if self.U_prev is not None:
            for i in range(self.nodes - 1):
                self.opti.set_initial(self.U_opt[i], self.U_prev[i+1])

    def init_solver(self, solver="fatrop"):
        if solver == "fatrop":
            opts = {
                "expand": True,
                "structure_detection": "auto",
                "debug": True,
            }
            opts["fatrop"] = {
                "print_level": 1,
                "tol": 1e-3,
                "mu_init": 1e-4,
                "warm_start_init_point": True,
                "warm_start_mult_bound_push": 1e-7,
                "bound_push": 1e-7,
            }
        elif solver == "ipopt":
            opts = {}
            opts["ipopt"] = {
                "print_level": 1,
                "tol": 1e-3,
            }
        else:
            raise ValueError("Unknown solver: ", solver)

        self.opti.solver(solver, opts)

    def solve(self, retract_all=True):
        try:
            self.sol = self.opti.solve()
            if self.sol.stats()["success"]:
                self.solve_time = self.sol.stats()["t_wall_total"]
                self._retract_opti_sol(retract_all)

            else:
                raise RuntimeError("Optimization problem did not solve successfully.")

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            self.sol = None

    def _retract_opti_sol(self, retract_all=True):
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]
        x_init = self.opti.value(self.x_init)

        for dx_sol, u_sol in zip(self.DX_prev, self.U_prev):
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.h_sol.append(np.array(x_sol[:6]).flatten())
            self.q_sol.append(np.array(x_sol[6:]).flatten())
            self.v_j_sol.append(np.array(u_sol[:self.nj]).flatten())
            self.forces_sol.append(np.array(u_sol[self.nj:]).flatten())

            if not retract_all:
                return
