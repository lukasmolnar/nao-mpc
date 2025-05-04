import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca


class Dynamics:
    def __init__(self, model, mass, foot_force_frames):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass
        self.foot_force_frames = foot_force_frames

        self.nq = model.nq
        self.nv = model.nv
        self.nj = model.nq - 7  # without base position and quaternion

    def state_integrate(self):
        """
        Input: x (with quaternion), dx (delta, without quaternion)
        Return: x_next (with quaternion)
        """
        x = ca.SX.sym("x", 6 + self.nq)
        dx = ca.SX.sym("dx", 6 + self.nv)

        h = x[:6]
        dh = dx[:6]
        h_next = h + dh

        q = x[6:]
        dq = dx[6:]
        q_next = cpin.integrate(self.model, q, dq)

        x_next = ca.vertcat(h_next, q_next)

        return ca.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])

    def state_difference(self):
        """
        Input: x0, x1 (with quaternions)
        Return: State difference (without quaternion)
        """
        x0 = ca.SX.sym("x0", 6 + self.nq)
        x1 = ca.SX.sym("x1", 6 + self.nq)

        h0 = x0[:6]
        q0 = x0[6:]
        h1 = x1[:6]
        q1 = x1[6:]

        dh = h1 - h0
        dq = cpin.difference(self.model, q0, q1)
        dx = ca.vertcat(dh, dq)

        return ca.Function("difference", [x0, x1], [dx], ["x0", "x1"], ["dx"])

    def com_dynamics(self):
        """
        Input: q, forces
        Return: h_dot (scaled by mass)
        """
        q = ca.SX.sym("q", self.nq)  # positions

        # Foot forces
        n_frames = len(self.foot_force_frames)
        foot_forces = [ca.SX.sym(f"f_foot_{i}", 3) for i in range(n_frames)]
        all_forces = ca.vertcat(*foot_forces)

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.centerOfMass(self.model, self.data)
        cpin.updateFramePlacements(self.model, self.data)

        # COM Dynamics
        g = np.array([0, 0, -9.81 * self.mass])
        dp_com = sum(foot_forces) + g
        dl_com = ca.SX.zeros(3)
        r_com = self.data.com[0]
        for idx, frame in enumerate(self.foot_force_frames):
            r_foot = self.data.oMf[frame].translation
            r_com_foot = r_foot - r_com
            dl_com += ca.cross(r_com_foot, foot_forces[idx])

        h_dot = ca.vertcat(dp_com, dl_com) / self.mass  # scale h by mass

        return ca.Function("com_dyn", [q, all_forces], [h_dot], ["q", "forces"], ["dh"])

    def base_vel_dynamics(self):
        h = ca.SX.sym("h", 6)  # COM momentum
        q = ca.SX.sym("q", self.nq)  # positions
        v_j = ca.SX.sym("v_j", self.nj)  # joint velocities

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        A = cpin.computeCentroidalMap(self.model, self.data, q)

        # Base velocity dynamics
        A_b = A[:, :6]
        A_j = A[:, 6:]
        A_b_inv = ca.inv(A_b)
        v_b = A_b_inv @ (h * self.mass - A_j @ v_j)  # scale h by mass

        return ca.Function("base_vel", [h, q, v_j], [v_b], ["h", "q", "v_j"], ["v_b"])
    
    def get_frame_position(self, frame):
        q = ca.SX.sym("q", self.nq)

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.updateFramePlacement(self.model, self.data, frame)
        pos = self.data.oMf[frame].translation

        return ca.Function("frame_pos", [q], [pos], ["q"], ["pos"])

    def get_frame_velocity(self, frame):
        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)

        # Pinocchio terms
        ref = pin.LOCAL_WORLD_ALIGNED
        cpin.forwardKinematics(self.model, self.data, q, v)
        frame_vel = cpin.getFrameVelocity(self.model, self.data, frame, ref).vector

        return ca.Function("frame_vel", [q, v], [frame_vel], ["q", "v"], ["vel"])
