from typing import List, Tuple, Dict, Union
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import collections


class JointCollection():

    def __init__(self, joints: Dict[any, np.ndarray], connections: List[Tuple[any, any]]=None) -> None:
        """Creates a new JointCollection.

        A Joint collection is a utility class intended to assist in calculating forward kinematics.
        Atm the word "joint" is a bit overloaded. It desribes both the individual positions and a combination of 
        3 positions to form an angle. 
        Currently the only type of joints explicity supported are "hinge" joints where they can only rotate 
        in the plane it forms. This joint can be totally described by the arccos angle and the axis 
        of rotation (normal vector).
        Args:
            joints (Dict[any, np.ndarray]): Mapping of keys to xyz positions.
            connections (List[Tuple[any, any]], optional): Connections between the joints. Defaults to None.
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
                self.lengths[joint1][joint2] = np.linalg.norm(self.pos[joint2] - self.pos[joint1])

    def add_connection(self, joint1, joint2):
        self.joints[joint1][joint2] = True
    
    def __getitem__(self, joints: Union[any, Tuple[any, any, any]]) -> Union[np.ndarray, R]:
        """Gets either a position or angle.

        If given a tuple of size 3, this will return the angle of the joint described by 
        the dictionary keys.
        If given a single dictionary key this will return the xyz position of the key.
        Args:
            joints (Union[any, Tuple[any, any, any]]): _description_

        Returns:
            Union[np.ndarray, R]: _description_
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

        # Save current angles
        angles = self.get_downstream_angles(joint3)
        og_pos3 = self.pos[joint3]

        # Calculate vec1/vec2 and the axis of rotation
        vec1 = self.pos[joint1] - self.pos[joint2]
        vec2 = self.pos[joint3] - self.pos[joint2]
        axis = np.cross(vec1, vec2)
        axis = axis / norm(axis)

        self.pos[joint3] = self.solve_for_pos(joint1, joint2, theta, axis, self.lengths[joint2][joint3])
        translation =  self.pos[joint3] - og_pos3 

        # Now we need to update positions of downstream joints
        # to be consistant with their og angles
        # angles is an ordered dict so dependecies should be in correct order
        for angle, (theta, _) in angles.items():
            joint1, joint2, joint3 = angle

            # We must translate vec2 by the same amount vec1 was in a prev iteration.
            # This is so that we can get a correct updated axis of rotation
            vec1 = self.pos[joint1] - self.pos[joint2]
            vec2 = self.pos[joint3] - self.pos[joint2] + translation
            new_axis = np.cross(vec1, vec2)
            new_axis = new_axis / norm(new_axis)

            self.pos[joint3] = self.solve_for_pos(joint1, joint2, theta, new_axis, self.lengths[joint2][joint3])

    def solve_for_pos(self, joint1: str, joint2: str, theta: int, axis: np.ndarray, length:int): 
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
        unit_parent = parent / np.linalg.norm(parent)
        q1 = R.from_quat([*unit_parent, 0])
        q2 = R.from_quat([*(axis * np.sin(theta / 2)), np.cos(theta/2)])
        q2_conj = R.from_quat(q2.as_quat() * np.array([-1, -1, -1, 1]))

        q3 = q2 * q1 * q2_conj

        return q3.as_quat()[:3] * length + pos2

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
                        vec1 = self.pos[parent] - self.pos[s]
                        vec2 = self.pos[child] - self.pos[s]
                        angle = self.find_arccos_angle(vec1, vec2)
                        axis = np.cross(vec1, vec2)
                        axis = axis / norm(axis)
                        angles[(parent, s, child)] = (angle, axis)

                    stack.append(child)
        return angles

    def find_arccos_angle(self, vec1, vec2):
        return np.arccos(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
    
    def get_all_angles(self):
        angles = {}
        for conn in self.connections:
            j2, j3 = conn
            if self.joint_parents[j2] is None:
                continue
            j1 = self.joint_parents[j2]
            angles[(j1, j2, j3)] = self[j1, j2, j3]
        return angles


