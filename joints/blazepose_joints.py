from typing import Dict
import joints as j
import numpy as np
from copy import deepcopy


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
        idxs[landmark] = idx_l

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

class BlazeposeHip(j.BallAndSocketJoint):

    def __init__(self): 
        super().__init__("hip", None, axis1=["right_hip", "left_hip"], axis2=np.array([0, 1, 0]),
                       angles_wanted={"forward-back": ("axis1", "axis2")})
    
    @property
    def joints(self):
        left_hip, right_hip = self.jc["left_hip"], self.jc["right_hip"]
        left_shoulder, right_shoulder = self.jc["left_shoulder"], self.jc["right_shoulder"]
        start = (left_hip + right_hip) / 2
        end = (left_shoulder + right_shoulder) / 2
        vec = end - start
        return vec
    
BLAZE_HIP = lambda: BlazeposeHip()

JOINTS = [BLAZE_LEFT_LEG, BLAZE_RIGHT_LEG, BLAZE_HIP,
          BLAZE_LEFT_SHOULDER, BLAZE_RIGHT_SHOULDER, BLAZE_LEFT_KNEE,
          BLAZE_RIGHT_KNEE, BLAZE_RIGHT_ELBOW, BLAZE_LEFT_ELBOW]


def joint_factory(jc: j.JointCollection):
    joints = []
    for joint in JOINTS:
        joint = joint()
        joint.set_joint_collection(jc)
        joints.append(joint)
    return joints


def get_angles(joints):
    return {k: v for d in joints for k, v in d.items()}


def frame_to_dict(frame):

    frame_dict = {}
    for landmark, idxs in LANDMARKS_TO_IDX.items():
        landmark_coords = frame[idxs]
        landmark_coords[1] *= -1
        landmark_coords[2] *= -1
        frame_dict[landmark] = landmark_coords
    return frame_dict


def dict_to_frame(d):
    frame = []
    for landmark in LANDMARKS:
        v = [*d[landmark], 1.0]
        v[1] *= -1
        v[2] *= -1
        frame.extend(v)
    return np.array(frame)


def frames_to_dicts(frames):
    return [frame_to_dict(frame) for frame in frames]


def fix_pose_avg(landmarks_avgs, lengths:Dict[str, Dict[str, int]]):
    graph = deepcopy(j.BLAZEPOSE_GRAPH)
    frame = j.frame_to_dict(landmarks_avgs)
    graph["root"] = ["left_hip", "right_hip"]

    lengths["root"] = {}
    lengths["root"]["left_hip"] = lengths["left_hip"]["right_hip"] / 2
    lengths["root"]["right_hip"] = lengths["left_hip"]["right_hip"] / 2

    new_frame = {k: np.copy(v) for k, v in frame.items()}
    bfs_fix_pose("left_shoulder", graph, frame, lengths, new_frame)
    bfs_fix_pose("right_shoulder", graph, frame, lengths, new_frame)
    bfs_fix_pose("left_hip", graph, frame, lengths, new_frame)
    bfs_fix_pose("right_hip", graph, frame, lengths, new_frame)

    return dict_to_frame(new_frame)

def bfs_fix_pose(root, graph, frame, lengths, new_frame):
    visited = set()
    parents = {root: None}

    queue = []
    queue.append((root, np.array([0, 0, 0])))
    while len(queue):
        joint, accum = queue[0]
        queue.pop(0)

        if joint not in visited:
            visited.add(joint)

        if joint not in graph.keys():
            continue

        for child in graph[joint]:
            if child not in visited:
                parents[child] = joint
                og = frame[child] - frame[joint]
                vec = og / np.linalg.norm(og)
                vec = vec * lengths[joint][child]
                diff = vec - og

                new_pos = vec + frame[joint] + accum
                new_frame[child] = new_pos

                queue.append((child, accum + diff))


def joint_collection_from_landmarks(landmarks):
    frame = frame_to_dict(landmarks)
    j.JointCollection(frame, BLAZEPOSE_CONNECTIONS)

def lengths_from_landmarks(landmarks):
    frame = frame_to_dict(landmarks)
    lengths = {landmark: {} for landmark in LANDMARKS}
    for conn in BLAZEPOSE_CONNECTIONS:
        joint1 = conn[0]
        joint2 = conn[1]
        lengths[joint1][joint2] = np.linalg.norm(frame[joint2] - frame[joint1])
    return lengths