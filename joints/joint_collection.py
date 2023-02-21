from typing import List, Tuple, Dict, Union
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import collections


def get_proj_to_plane(vec, normal):
    normal = normal / norm(normal)
    vec = vec / norm(vec)
    proj = np.dot(vec, normal) * normal
    return proj


def proj_to_plane(vec, axis):
    vec = vec / norm(vec)
    return vec - get_proj_to_plane(vec, axis)


def to_quat(axis, theta):
    print(theta)
    print([*(axis * np.sin(theta / 2)), np.cos(theta/2)])
    return R.from_quat([*(axis * np.sin(theta / 2)), np.cos(theta/2)])


def p_to_vec(a, b, c):
    return a - b, c - b


def rotate_vector(vec: np.ndarray, axis: np.ndarray, theta: int):
    axis = axis / norm(axis)
    q1 = R.from_quat([*vec, 0])
    q2 = to_quat(axis, theta)
    q2_conj = R.from_quat(q2.as_quat() * np.array([-1, -1, -1, 1]))

    q3 = q2 * q1 * q2_conj

    return q3.as_quat()[:3] * norm(vec)


def arccos_angle(vec1, vec2):
    normalized_dot = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    if normalized_dot > 1:
        normalized_dot = 1
    if normalized_dot < -1:
        normalized_dot = -1
    return np.arccos(normalized_dot)


def find_signed_angle(vec1, vec2, axis=None):
    if axis is None:
        axis = np.cross(vec1, vec2)
    arccos_theta = arccos_angle(vec1, vec2)
    pos = rotate_vector(vec2, axis, arccos_theta)
    neg = rotate_vector(vec2, axis, -1 * arccos_theta)
    vs_pos, vs_neg = arccos_angle(vec1, pos), arccos_angle(vec1, neg)

    if vs_pos < vs_neg:
        return arccos_theta
    else:
        return -1 * arccos_theta


def set_angle_between_vectors(vec1, vec2, theta):
    set_angle_about_axis(vec1, vec2, np.cross(vec1, vec2, theta))


def set_angle_about_axis(vec1, vec2, axis, theta):

    # axis must describe the correct handness of the rotation
    axis = axis / norm(axis)
    curr_theta = arccos_angle(vec1, vec2)
    theta_to_rotate = theta - curr_theta
    return rotate_vector(vec2, axis, theta_to_rotate)


def set_about_lcs(vec_to_rotate, rot_axis, axis2, theta):
    assert np.isclose(np.dot(rot_axis, axis2), 0, atol=1e15)
    projected_vec = proj_to_plane(vec_to_rotate, rot_axis)

    curr_theta = find_signed_angle(projected_vec, axis2, rot_axis)
    theta_to_rotate = theta - curr_theta

    return rotate_vector(projected_vec, rot_axis, theta_to_rotate)


class JointCollection():

    def __init__(self, joints: Dict[any, np.ndarray], connections: List[Tuple[any, any]]) -> None:
        """Creates a new JointCollection.

        A Joint collection is a utility class intended to assist in calculating forward kinematics.
        Atm the word "joint" is a bit overloaded. It desribes both the individual positions and a combination of 
        3 positions to form an angle. 
        Currently the only type of joints explicity supported are "hinge" joints where they can only rotate 
        in the plane it forms. This joint can be totally described by the arccos angle and the axis 
        of rotation (normal vector).
        Args:
            joints (Dict[any, np.ndarray]): Mapping of keys to xyz positions.
            connections (List[Tuple[any, any]], optional): Connections between the joints.
        """
        self.joints = {joint: set() for joint in joints.keys()}
        self.connections = connections
        self.joint_parents = {joint: None for joint in joints.keys()}
        self.pos = {joint: pos for joint, pos in joints.items()}

        self.lengths = {joint: dict() for joint in joints.keys()}

        if connections is not None:
            for conn in connections:
                joint1 = conn[0]
                joint2 = conn[1]
                self.joints[joint1].add(joint2)
                self.joint_parents[joint2] = joint1
                self.lengths[joint1][joint2] = np.linalg.norm(
                    self.pos[joint2] - self.pos[joint1])

    def __getitem__(self, joints: Union[any, Tuple[any, any, any]]) -> Union[np.ndarray, R]:
        """Gets either a position or angle.

        If given a tuple of size 3, this will return the quaternion of the joint described by 
        the dictionary keys.
        If given a single dictionary key this will return the xyz position of the key.
        Args:
            joints (Union[any, Tuple[any, any, any]]): A single dict key or a tuple of 3.

        Returns:
            Union[np.ndarray, R]: The position or quaternion.
        """
        if joints in self.pos.keys():
            return self.pos[joints]

        if isinstance(joints, collections.Sequence):
            assert len(joints) == 3
            vec1 = self.pos[joints[0]] - self.pos[joints[1]]
            vec2 = self.pos[joints[2]] - self.pos[joints[1]]

            xyz = np.cross(vec1, vec2)
            w = np.sqrt(norm(vec1)**2 * norm(vec2)**2) + np.dot(vec1, vec2)
            return R.from_quat([*xyz, w])

    def rotate_about_axis(self, a, b, axis, theta):
        # Save current angles
        angles = self.get_downstream_angles(b)

        axis = axis / norm(axis)
        vec_to_rotate = self.pos[b] - self.pos[a]

        self.pos[b] = rotate_vector(vec_to_rotate, axis, theta) + self.pos[a]

        # Now we need to update positions of downstream joints
        # to be consistant with their og angles
        # angles is an ordered dict so dependecies should be in correct order
        for angle, _ in angles.items():
            joint1, joint2, joint3 = angle

            vec = self.pos[joint3] - self.pos[a]
            relative = rotate_vector(vec, axis, theta)
            self.pos[joint3] = relative + self.pos[a]

    def set_about_lcs(self, a, b, rot_axis, axis2, theta):
        assert np.isclose(np.dot(rot_axis, axis2), 0, atol=1e15)
        vec = proj_to_plane(self.pos[b] - self.pos[a], rot_axis)

        curr_theta = find_signed_angle(vec, axis2, rot_axis)
        theta_to_rotate = theta - curr_theta

        self.rotate_about_axis(a, b, rot_axis, theta_to_rotate)

    def set_about_axis(self, a, b, c, axis, theta):

        axis = axis / norm(axis)
        curr_theta = np.arccos(self[a, b, c].as_quat()[-1]) * 2
        theta_to_rotate = theta - curr_theta
        self.rotate_about_axis(b, c, axis, theta_to_rotate)

    def __setitem__(self, joints: Tuple[str], theta: int):
        """Updates the given angle and all downstream positions.

        Ensures after updating the given joint to the needed angle
        all downstream angles remain the same after the rotation.
        Args:
            joints (Tuple[str]): The size 3 tuple of the 3 positions that will make up the angle.
                joints = (j1, j2, j3). vec1 (parent) = j2 -> j1. vec2 (child to change) = j2 -> j3.
            theta (int): Angle to set joint to. In radians.
        """
        joint1, joint2, joint3 = joints

        # Calculate vec1/vec2 and the axis of rotation
        vec1 = self.pos[joint1] - self.pos[joint2]
        vec2 = self.pos[joint3] - self.pos[joint2]
        axis = np.cross(vec1, vec2)
        axis = axis / norm(axis)
        self.set_about_axis(joint1, joint2, joint3, axis, theta)

    def solve_for_joint_pos(self, joint1: str, joint2: str, theta: int, axis: np.ndarray, length: int):
        """Solves for position using quaternions.

        Given we have a system with 3 points j1, j2, j3 which make vectors 
        (j2 -> j1) and (j2 -> j3), we want to solve for j3. We solve for j3
        given the angle (theta) and the axis of rotation which will be normal
        to the vector system. 

        Args:
            joint1 (str): First joints
            joint2 (str): Second joint
            theta (int): Angle between joints (j2 -> j1) and (j2 -> j3) with respect to the given axis
            axis (np.ndarray): Normal vector to the vectors.
            length (int): Length of desired vector

        Returns:
            np.ndarray: shape (3) numpy array containing the xyz of the solved position.
        """
        pos1 = self.pos[joint1]
        pos2 = self.pos[joint2]
        parent = pos1 - pos2
        parent = parent / norm(parent)
        return rotate_vector(parent, axis, theta) * length + pos2

    def get_downstream_angles(self, s) -> Dict[Tuple[str, str, str], Tuple[int, np.ndarray]]:
        """DFS to get all downstream angles.

        The output starts with the root at the center of the joint. So the first joint is
        (parent(s), s, child(s))
        Args:
            s (any): dictionary key corresponding to the root joint.

        Returns:
            Dict[Tuple[any, any, any], Tuple[int, np.ndarray]]: Ordered dictionary containing the
            mapping of joints to an angle and normal vector (axis of rotation).
        """
        visited = set()
        angles = {}

        stack = []
        stack.append(s)
        while len(stack):
            s = stack[-1]
            stack.pop()

            if s not in visited:
                visited.add(s)

            for child in self.joints[s]:
                if child not in visited:
                    parent = self.joint_parents[s]

                    # parent is only none if it is the waist node
                    if parent is not None:
                        angles[(parent, s, child)] = self[parent, s, child]

                    stack.append(child)
        return angles

    def get_all_angles(self):
        angles = {}
        for conn in self.connections:
            j2, j3 = conn
            if self.joint_parents[j2] is None:
                continue
            j1 = self.joint_parents[j2]
            angles[(j1, j2, j3)] = self[j1, j2, j3]
        return angles


def Joint():

    def __init__(self, name, jc: JointCollection, joints: List = None):
        self.name = name
        self.jc = jc

        if isinstance(joints, collections.Sequence):
            assert len(joints) == 3
            self.joints = None

    def eval(self):
        if self.joints is not None:
            return self.jc[self.joints]
        raise NotImplementedError
