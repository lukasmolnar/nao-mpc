from os.path import dirname, abspath

import numpy as np
import pinocchio as pin
import casadi as ca
from pinocchio.robot_wrapper import RobotWrapper


def rad(deg):
    return np.deg2rad(deg)


class NAO:
    def __init__(self):
        urdf_path = "nao_description/urdf/nao_simple.urdf"
        urdf_dir = dirname(abspath(urdf_path))
        joint_model = pin.JointModelFreeFlyer()

        # Lock joint indices
        lock_joints = [
            2, 3,  # head yaw, pitch
            10, 11, 12, 13, 14, 15,  # left arm
            22, 23, 24, 25, 26, 27,  # right arm
        ]

        # Nominal configuration (with all joints)
        self.q_nom = np.array([
            0, 0, 0.313, 0, 0, 0, 1,                  # base position + quaternion
            0, 0,                                     # head
            0, 0, rad(-21.3), rad(49), rad(-27.6), 0, # left leg
            rad(90), rad(5), 0, 0, rad(-90), 0,       # left arm 
            0, 0, rad(-21.3), rad(49), rad(-27.6), 0, # right leg
            rad(90), rad(-5), 0, 0, rad(90), 0,       # right arm
        ])

        # Nominal configuration (without locked joints)
        self.q0 = np.array([
            0, 0, 0.313, 0, 0, 0, 1,                  # base position + quaternion
            0, 0, rad(-21.3), rad(49), rad(-27.6), 0, # left leg joints
            0, 0, rad(-21.3), rad(49), rad(-27.6), 0, # right leg joints
        ])

        # Build robot
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [urdf_dir], joint_model)
        self.robot = self.robot.buildReducedRobot(lock_joints, self.q_nom)
        self.model = self.robot.model
        self.data = self.robot.data

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

    def set_gait_sequence(self, gait_type, gait_nodes, step_height, dt):
        self.gait_sequence = GaitSequence(gait_type, gait_nodes, step_height, dt)
        self.foot_force_frames = [self.model.getFrameId(f) for f in self.gait_sequence.foot_force_frames]
        self.foot_center_frames = [self.model.getFrameId(f) for f in self.gait_sequence.foot_center_frames]
        self.nf = 3 * len(self.foot_force_frames)  # number of total forces


class GaitSequence:
    def __init__(self, gait_type="walk", gait_nodes=20, step_height=0.02, dt=0.02):
        self.foot_force_frames = ["l_heel_in", "l_heel_out", "l_toe_in", "l_toe_out",
                                  "r_heel_in", "r_heel_out", "r_toe_in", "r_toe_out"]
        self.foot_center_frames = ["l_sole", "r_sole"]
        self.gait_nodes = gait_nodes
        self.step_height = step_height
        self.dt = dt

        # Idx 0: left foot, 1: right foot
        self.contact_schedule = np.ones((2, gait_nodes))  # default: in contact
        self.bezier_schedule = np.zeros((2, gait_nodes))  # default: phase 0

        if gait_type == "walk":
            self.N = gait_nodes // 2
            self.n_contacts = 2
            for i in range(gait_nodes):
                bezier_phase = i % self.N / (self.N - 1)  # normalize to [0, 1]
                if i < self.N:
                    # Left foot swing
                    self.contact_schedule[0, i] = 0
                    self.bezier_schedule[0, i] = bezier_phase
                else:
                    # Right foot swing
                    self.contact_schedule[1, i] = 0
                    self.bezier_schedule[1, i] = bezier_phase

        elif gait_type == "stand":
            self.N = gait_nodes
            self.n_contacts = 4

        else:
            raise ValueError(f"Gait: {gait_type} not supported")

    def shift_contact_schedule(self, shift_idx):
        shift_idx %= self.gait_nodes
        return np.roll(self.contact_schedule, -shift_idx, axis=1)

    def shift_bezier_schedule(self, shift_idx):
        shift_idx %= self.gait_nodes
        return np.roll(self.bezier_schedule, -shift_idx, axis=1)

    def get_bezier_vel(self, bezier_phase):
        # Bezier curve velocity
        T = self.N * self.dt  # period in seconds
        vel_z = ca.if_else(
            bezier_phase < 0.5,
            self.cubic_bezier_derivative(0, self.step_height, 2 * bezier_phase),
            self.cubic_bezier_derivative(self.step_height, 0, 2 * bezier_phase - 1)
        ) * 2 / T

        return vel_z

    def cubic_bezier_derivative(self, p0, p1, idx):
        return 6 * idx * (1 - idx) * (p1 - p0)
