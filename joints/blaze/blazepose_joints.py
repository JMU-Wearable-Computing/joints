from typing import Dict
import joints as j
import numpy as np
from typing import List, Union, Dict, Tuple
from scipy.spatial.transform import Rotation as R


LANDMARKS = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", 
                "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
                "right_ear", "mouth_left", "mouth_right", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                "right_wrist", "left_pinky", "right_pinky", "left_index",
                "right_index", "left_thumb", "right_thumb", "left_hip", 
                "right_hip", "left_knee", "right_knee", "left_ankle",
                "right_ankle", "left_heel", "right_heel", "left_foot_index",
                "right_foot_index"]


# I coppied connections from blazepose instead of importing them to reduce dependencies
# I made some changes to make the connects for a useful directed graph
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose_connections.py
CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (23, 11), (24, 12), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

BLAZEPOSE_CONNECTIONS = [(LANDMARKS[c1], LANDMARKS[c2]) for c1, c2 in CONNECTIONS]

tmp_dict = {v1: [] for (v1, v2) in BLAZEPOSE_CONNECTIONS}
for (v1, v2) in BLAZEPOSE_CONNECTIONS:
    tmp_dict[v1].append(v2)
BLAZEPOSE_GRAPH = tmp_dict 


def _make_landmarks_dict(z=True, vis=False):
    
    idxs = {}
    for i, landmark in enumerate(LANDMARKS):
        idx = i * 4
        idx_l = [idx, idx + 1]
        if z:
            idx_l.append(idx + 2)
        if vis:
            idx_l.append(idx + 3)
        idxs[landmark] = np.array(idx_l)

    return idxs


LANDMARKS_TO_IDX = _make_landmarks_dict()


BLAZE_LEFT_KNEE = lambda: j.Joint("left_knee", ["left_hip", "left_knee", "left_ankle"])
BLAZE_RIGHT_KNEE = lambda: j.Joint("right_knee", ["right_hip", "right_knee", "right_ankle"])

BLAZE_LEFT_ELBOW = lambda: j.Joint("left_elbow", ["left_shoulder", "left_elbow", "left_wrist"])
BLAZE_RIGHT_ELBOW = lambda: j.Joint("right_elbow", ["right_shoulder", "right_elbow", "right_wrist"])

BLAZE_LEFT_SHOULDER = lambda: j.BallAndSocketJoint("left_shoulder",
                                     ["left_shoulder", "left_elbow"],
                                     axis1=["right_shoulder", "left_shoulder"],
                                     # Flipped this order to start zero angle with arms down
                                     axis2=["left_shoulder", "left_hip"], 
                                     axis3="-",
                                     angles_wanted={"up-down": ("axis3", "axis2"), "forward-back": ("axis2", "axis1")})

BLAZE_RIGHT_SHOULDER = lambda: j.BallAndSocketJoint("right_shoulder",
                                     ["right_shoulder", "right_elbow"],
                                     axis1=["left_shoulder", "right_shoulder"],
                                     # Flipped this order to start zero angle with arms down
                                     axis2=["right_shoulder", "right_hip"], 
                                     axis3="-",
                                     angles_wanted={"up-down": ("axis3", "axis2"), "forward-back": ("-axis2", "axis1")})


BLAZE_LEFT_LEG = lambda: j.BallAndSocketJoint("left_leg",
                                     ["left_hip", "left_knee"],
                                     axis1=["right_hip", "left_hip"],
                                     # Flipped this order to start zero angle with arms down
                                     axis2=["left_shoulder", "left_hip"], 
                                     axis3="-",
                                     angles_wanted={"left-right": ("axis3", "axis2"), "forward-back": ("-axis1", "axis2")})

BLAZE_RIGHT_LEG = lambda: j.BallAndSocketJoint("right_leg",
                                     ["right_hip", "right_knee"],
                                     axis1=["left_hip", "right_hip"],
                                     # Flipped this order to start zero angle with arms down
                                     axis2=["right_shoulder", "right_hip"], 
                                     axis3="-",
                                     angles_wanted={"left-right": ("axis3", "axis2"), "forward-back": ("axis1", "axis2")})

class BlazeposeBack(j.BallAndSocketJoint):

    def __init__(self): 
        super().__init__("back", None, axis1=["right_hip", "left_hip"], axis2=np.array([0, 1, 0]),
                       angles_wanted={"forward-back": ("axis1", "axis2")})
    
    @property
    def joints(self):
        left_hip, right_hip = self.jc["left_hip"], self.jc["right_hip"]
        left_shoulder, right_shoulder = self.jc["left_shoulder"], self.jc["right_shoulder"]
        start = (left_hip + right_hip) / 2
        end = (left_shoulder + right_shoulder) / 2
        vec = end - start
        return vec
    
BLAZE_BACK = lambda: BlazeposeBack()

JOINTS = [BLAZE_LEFT_LEG, BLAZE_RIGHT_LEG, BLAZE_BACK,
          BLAZE_LEFT_SHOULDER, BLAZE_RIGHT_SHOULDER, BLAZE_LEFT_KNEE,
          BLAZE_RIGHT_KNEE, BLAZE_RIGHT_ELBOW, BLAZE_LEFT_ELBOW]



def joint_factory(jc: j.JointCollection=None) -> List[j.Joint]:
    """Makes new joint objects.

    Args:
        jc (j.JointCollection, optional): If specified the joints
        will have this jc set. Defaults to None.

    Returns:
        List[j.Joint]: The list of created joints.
    """
    joints = []
    for joint in JOINTS:
        joint = joint()
        if jc is not None:
            joint.set_joint_collection(jc)
        joints.append(joint)
    return joints


def set_joint_collection(joints: List[j.Joint], jc: j.JointCollection) -> None:
    """Sets the given joint collection on the list of joints.

    Args:
        joints (List[j.Joint]): List of joints to set the jc on.
        jc (j.JointCollection): JC to set the joints to.
    """
    for joint in joints:
        joint.set_joint_collection(jc)


def get_angles(joints: List[j.Joint], degrees: bool=False) -> Dict[any, float]:
    """Calls .angle(degrees) on each joint and returns a merged dict.

    Args:
        joints (List[j.Joint]): The joint list.
        degrees (bool, optional): If the angles should be in degrees.
        If false it will be in radians. Defaults to False.

    Returns:
        Dict[any, float]: Merged dictionary of joints angles.
    """
    return {k: v for d in joints for k, v in d.angle(degrees).items()}


def get_rotations(joints: List[j.Joint]) -> Dict[any, R]:
    """Calls .rotation() of each joints and returns a merged dict.

    Args:
        joints (List[j.Joint]): The list of joints to obtain the
        rotations.

    Returns:
        Dict[any, R]: THe merged dictionary of joint rotations
    """
    return {k: v for d in joints for k, v in d.rotation().items()}


def frame_to_dict(frame: np.array) -> Dict[str, np.ndarray]:
    """Converts a single frame to a dictionary landmarks.

    Args:
        frame (np.array): Frame to convert to a dictionary.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping landmarks
        to coordinates.
    """
    frame_dict = {}
    for landmark, idxs in LANDMARKS_TO_IDX.items():
        landmark_coords = frame[idxs]
        landmark_coords[1] *= -1
        landmark_coords[2] *= -1
        frame_dict[landmark] = landmark_coords
    return frame_dict


def frames_to_dicts(frames: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Converts a list of frames to a list of dicts.

    Args:
        frames (List[np.ndarray]): 

    Returns:
        List[Dict[str, np.ndarray]]: List of converted frames.
    """
    return [frame_to_dict(frame) for frame in frames]


def dict_to_frame(d: Dict[str, np.ndarray]) -> np.ndarray:
    """Converts a dictionary back to frame.

    Args:
        d (Dict[str, np.ndarray]): Dictionary to convert.

    Returns:
        np.ndarray: The converted frame.
    """
    frame = []
    for landmark in LANDMARKS:
        v = [*d[landmark], 1.0]
        v[1] *= -1
        v[2] *= -1
        frame.extend(v)
    return np.array(frame)


def jc_from_landmarks(landmarks: j.Landmarks) -> Union[j.JointCollection, List[j.JointCollection]]:
    """_summary_

    Args:
        landmarks (j.Landmarks): Landmarks to use to make joint collection. 

    Returns:
        Union[j.JointCollection, List[j.JointCollection]]: _description_
    """
    landmarks = np.array(landmarks)

    if (len(landmarks.shape) > 1
        and 1 not in landmarks.shape):
        frames = frames_to_dicts(landmarks)
        return [j.JointCollection(frame, BLAZEPOSE_CONNECTIONS) for frame in frames]
    elif (len(landmarks.shape) == 1
            and landmarks.size == len(LANDMARKS) * 4):
        frame = frame_to_dict(landmarks)
        return j.JointCollection(frame, BLAZEPOSE_CONNECTIONS)
    else:
        raise Exception(f"""Landmarks passed as numpy array have 
                            incompatiple shape {landmarks.shape}.
                            They must be [{len(LANDMARKS) * 4}]
                            or [n_frames, {len(LANDMARKS) * 4}]""")


def lengths_from_landmarks(landmarks: j.Landmarks) -> Dict[str, Dict[str, float]]:
    """Converts landmarks to lengths between connections.

    Args:
        landmarks (j.Landmarks): Landmarks to extract lengths from.

    Returns:
        Dict[str, float]: Dictionary mapping connections to lengths
    """
    frame = frame_to_dict(landmarks)
    lengths = {landmark: {} for landmark in LANDMARKS}
    for conn in BLAZEPOSE_CONNECTIONS:
        joint1 = conn[0]
        joint2 = conn[1]
        lengths[joint1][joint2] = np.linalg.norm(frame[joint2] - frame[joint1])
    return lengths


def get_all_angles_from_landmarks(landmarks: j.Landmarks, degrees: bool=False) \
    -> Union[Dict[str, float], Dict[str, List[float]]]:
    """Gets all angles from given landmarks.
    If given a list of landmarks then it will return a dict of lists of angles 
    mapping angles name to a list of angles from their respecitive landmarks.
    If given a single landmark then it will return a dict of angle names to
    a single float.

    Args:
        landmarks (j.Landmarks): Landmarks to get angles from
        degrees (bool, optional): If to use degrees or radians. Defaults to False.

    Returns:
        Union[Dict[str, float], Dict[str, List[float]]]: Dict mapping angle names
        to angle.
    """
    jcs = jc_from_landmarks(landmarks)
    joints = joint_factory()

    if isinstance(jcs, List):
        set_joint_collection(joints, jcs[0])
        angles = {k:[v] for k, v in get_angles(joints, degrees).items()}
        for jc in jcs[1:]:
            set_joint_collection(joints, jc)
            curr = get_angles(joints, degrees)
            for angle in angles.keys():
                angles[angle].append(curr[angle])
        return angles
    else:
        set_joint_collection(joints, jcs)
        return get_angles(joints, degrees)


def joints_from_landmarks(landmarks: j.Landmarks, return_jc=False) \
    -> Union[List[List[j.Joint]], Tuple[List[List[j.Joint]], List[j.JointCollection]]]:
    """Returns a list of joints from each landmark.

    Args:
        landmarks (j.Landmarks): Landmarks to make joints from. 
        return_jc (bool, optional): If to return the joint collectsions for each joints.
        Defaults to False.

    Returns:
        Union[List[List[j.Joint]], Tuple[List[List[j.Joint]], List[j.JointCollection]]]: 
        The list of joints from each landmark and optionally the JC.
    """
    jcs = jc_from_landmarks(landmarks)

    if isinstance(jcs, List):
        all_joints = []
        for jc in jcs:
            joints = joint_factory()
            set_joint_collection(joints, jc)
            all_joints.append(all_joints)
        if return_jc:
            return all_joints, jcs
        return all_joints

    joints = joint_factory()
    set_joint_collection(joints, jcs)
    if return_jc:
        return [joints], [jcs]
    return [joints]

