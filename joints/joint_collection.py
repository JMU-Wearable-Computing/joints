from typing import List, Tuple, Dict, Union
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from collections.abc import Sequence
import math
import copy

def get_proj_to_plane(vec, normal):
    normal = normal / norm(normal)
    vec = vec / norm(vec)
    proj = np.dot(vec, normal) * normal
    return proj


def proj_to_plane(vec, axis):
    vec = vec / norm(vec)
    return vec - get_proj_to_plane(vec, axis)


def to_quat(axis, theta):
    return R.from_quat([*(axis * np.sin(theta / 2)), np.cos(theta/2)])


def p_to_vec(a, b):
    return b - a


def rotate_scipy (vec: np.ndarray, rotation):
    """
    Given a position vector and and scipy rotation object,
        rotate that position by that rotation
    _____________________________________
    Args:
        vec:
            position (xyz) vector
        rotation:
            an scipy rotation object representing some rotation
    ______________________________________
    Returns:
        the position vector rotated by rotation
    """
    return np.dot(vec, rotation.as_matrix())
    q1 = R.from_quat([*vec, 0])
    q2 = rotation
    q2_conj = R.from_quat(q2.as_quat() * np.array([-1, -1, -1, 1]))

    q3 = q2 * q1 * q2_conj

    return q3.as_quat()[:3] * norm(vec)

def rotate_vector(vec: np.ndarray, axis: np.ndarray, theta: int):
    axis = axis / norm(axis)
    q1 = R.from_quat([*vec, 0])
    q2 = to_quat(axis, theta)
    q2_conj = R.from_quat(q2.as_quat() * np.array([-1, -1, -1, 1]))

    q3 = q2 * q1 * q2_conj

    return q3.as_quat()[:3] * norm(vec)


def difference_of_rotations (first_rot, second_rot):
    return second_rot * first_rot.inv()

def euler_difference (first_rot, second_rot):
    euler1 = first_rot.as_euler('zyx')
    euler2 = second_rot.as_euler('zyx')
    angular_difference = R.from_euler('zyx',np.arctan2(np.sin(euler1-euler2), np.cos(euler1-euler2)))
    return angular_difference


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
    else:
        vec1 = proj_to_plane(vec1, axis)
        vec2 = proj_to_plane(vec2, axis)
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

    def __init__(self, joint_pos: Dict[any, np.ndarray], connections: List[Tuple[any, any]]) -> None:
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
        self.connections = connections
        self.pos = joint_pos

        self._joint_children = None
        self._joint_parents = None
    
    @property
    def joint_children(self):
        if self._joint_children is None:
            self._joint_children = {joint: set() for joint in self.pos.keys()}
            for conn in self.connections:
                self._joint_children[conn[0]].add(conn[1])
        return self._joint_children

    @property
    def joint_parents(self):
        if self._joint_parents is None:
            self._joint_parents = {joint: None for joint in self.pos.keys()}
            for conn in self.connections:
                self._joint_parents[conn[1]] = conn[0]
        return self._joint_parents


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

        if isinstance(joints, Sequence):
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
    def get_euc_distance(self, a, b):
        return math.sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
     

    def rotate_with_scipy(self, a, b, rotation, dict_name=None):
        # Save current angles
        angles = self.get_downstream_angles(b)

        # get 
        old_pos = copy.deepcopy(self.pos)
        vec_to_rotate = old_pos[b] - old_pos[a]

        self.pos[b] = rotate_scipy(vec_to_rotate, rotation) + old_pos[a]
        
        #print (f"rotating {b} around {a}")

        # Using the saved old positions t get base joint distance, and and translate
        # real position of previous joint by difference of the base model      

        for angle, _ in angles.items():
            joint1, joint2, joint3 = angle
            # this SHOULD NOT BE AN 0 VECTOR!
            joint_at_origin = old_pos[joint3] - old_pos[joint2]
            if joint_at_origin[0] == 0 and joint_at_origin[1] == 0 and joint_at_origin[2] == 0:
                print (f"ERROR: when attempting to rotate {b} around {a}. The current joint is {joint3}\n     Something probably went wrong, encountered a 0 position vector")
                print (f"   Dict name is: {dict_name}")
                continue
            else:
                b_distance = self.get_euc_distance(self.pos[joint2], self.pos[joint3])
                #print (f"   Before Distance of  {joint2} to {joint3} is { } ")
                relative_movement = rotate_scipy(joint_at_origin, rotation)
                self.pos[joint3] = relative_movement + self.pos[joint2]
                a_distance = self.get_euc_distance(self.pos[joint2], self.pos[joint3])
                #print (f"   After Distance of  {joint2} to {joint3} is {self.get_euc_distance(self.pos[joint2], self.pos[joint3])}")
                #if abs(a_distance - b_distance) > .1:
                #    print (f"   Distance difference before and after rotation for {joint3} around {a} is {a_distance - b_distance}")
            #print (f"vec is {vec}")
            #print (f"rotation is {rotation.as_euler('zyx',degrees=True)}")
        #print ("___________________")


    def set_about_lcs(self, a, b, rot_axis, axis2, theta, proj_axis=False):
        if proj_axis:
            axis2 = proj_to_plane(axis2, rot_axis)
        else:
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
        """BFS to get all downstream angles.

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
            s = stack[0]
            stack.pop(0)

            if s not in visited:
                visited.add(s)

            for child in self.joint_children[s]:
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


class Joint():

    def __init__(self, name, joints: Tuple = None, jc: JointCollection=None):
        self.name = name
        self.jc = jc

        if joints is not None:
            assert isinstance(joints, Sequence)
            assert len(joints) == 3
            self.joints = joints
    
    def p_to_vec(self, a, b):
        return self.jc[b] - self.jc[a]
    
    def angle(self, degrees=False):
        j0 = self.jc[self.joints[0]]
        j1 = self.jc[self.joints[1]]
        j2 = self.jc[self.joints[2]]
        vec1 = j0 - j1
        vec2 = j2 - j1

        angle = arccos_angle(vec1, vec2)
        if degrees:
            angle = (angle * 180) / np.pi
        return {self.name: angle}

    def rotation(self):
        return {self.name: self.jc[tuple(self.joints)]}

    def set_angle(self, theta):
        self.jc[tuple(self.joints)] = theta
    
    def set_joint_collection(self, jc: JointCollection):
        self.jc = jc


class BallAndSocketJoint(Joint):

    def __init__(self, name, joints, axis1, axis2, axis3="+", jc=None,
                 angles_wanted={"forward-back": ("axis2", "axis1"), "up-down": ("axis3", "axis2")}):
        super().__init__(name, jc=jc)

        if joints is not None:
            assert len(joints) == 2, "Only pass two joints for a ball and socket joints."
        self._joints = joints
        self._axis1 = axis1
        self._axis2 = axis2
        self._axis3 = axis3
        self.angles_wanted = angles_wanted
        self.axis_map = {"axis1": lambda: self.axis1,
                         "axis2": lambda: self.axis2, 
                         "axis3": lambda: self.axis3,
                         "-axis1": lambda: -1 * self.axis1,
                         "-axis2": lambda: -1 * self.axis2, 
                         "-axis3": lambda: -1 * self.axis3
                        }
    @property
    def joints(self):
        return self.standardize_vec(self._joints)

    @property
    def axis1(self):
        axis1 = self.standardize_vec(self._axis1)
        return axis1 / norm(axis1)

    @property
    def axis2(self):
        axis2 =  self.standardize_vec(self._axis2)
        return axis2 / norm(axis2)


    @property
    def axis3(self):
        if self._axis3 == "+":
            axis3 = np.cross(self.p_to_vec(*self._axis1),
                            self.p_to_vec(*self._axis2))
        elif self._axis3 == "-":
            axis3 = -1 * np.cross(self.p_to_vec(*self._axis1),
                                 self.p_to_vec(*self._axis2))
        else:
            axis3 = self.standardize_vec(self._axis3)
        return axis3 / norm(axis3)

    def standardize_vec(self, vec):
        if (isinstance(vec, Sequence)
            and isinstance(vec[0], str)
            and len(vec) == 2):
            return self.p_to_vec(*vec) 
        elif (isinstance(vec, Sequence) or isinstance(vec, np.ndarray)) and len(vec) == 3:
            return vec
        else:
            raise Exception(f"vector not in correct fromat: {vec}")
    
    def axis_angle(self, rot_axis, axis2, degrees=False):
        angle = find_signed_angle(self.joints, axis2, axis=rot_axis)
        if degrees:
            angle = (angle * 180) / np.pi
        return angle

    def axis_rotation(self, rot_axis, axis2):
        angle = find_signed_angle(self.joints, axis2, axis=rot_axis)
        return to_quat(rot_axis, angle)
    
    def angle(self, degrees=False):
        def helper(rot_axis, start_vec):
            return self.axis_angle(self.axis_map[rot_axis](),
                                   self.axis_map[start_vec](),
                                   degrees)
        return {f"{self.name}_{k}": helper(rot_axis, start_vec)
                for k, (rot_axis, start_vec) in self.angles_wanted.items()}
    
    def rotation(self):
        return {f"{self.name}_{k}": self.axis_rotation(self.axis_map[rot_axis](), self.axis_map[start_vec]())
                for k, (rot_axis, start_vec) in self.angles_wanted.items()}
    