import mediapipe as mp
import cv2
from os import path
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from google.protobuf import text_format
import joints as j
from copy import deepcopy
from typing import List, Dict, Tuple, Union
from collections.abc import Generator


def write_frame(landmark, wlandmark, landmark_file=None, wlandmark_file=None):
    landmark = ",".join(map(str, landmark))[:-1] + "\n"
    landmark_file.write(landmark)

    wlandmark = ",".join(map(str, wlandmark))[:-1] + "\n"
    wlandmark_file.write(wlandmark)


def unpack_frame(results):
    landmarks = results.pose_landmarks
    wlandmarks = results.pose_world_landmarks

    # if we actually have a skeleton
    if landmarks is not None:
        landmarks = landmarks.landmark
    if wlandmarks is not None:
        wlandmarks = wlandmarks.landmark

    curr_frame_raw = []
    curr_wframe_raw = []
    # iterate through all markers and write out x, y, z, and visibility
    for marker_index in range(0, len(landmarks)):

        x = landmarks[marker_index].x
        y = landmarks[marker_index].y
        z = landmarks[marker_index].z
        vis = landmarks[marker_index].visibility
        curr_frame_raw.extend([x, y, z, vis])

        x = wlandmarks[marker_index].x
        y = wlandmarks[marker_index].y
        z = wlandmarks[marker_index].z
        vis = wlandmarks[marker_index].visibility
        curr_wframe_raw.extend([x, y, z, vis])

    return curr_frame_raw, curr_wframe_raw


def capture(window_name: str="blaze",
            file_name: str="last_recording",
            model_complexity: int=1,
            return_image: bool=False,
            stream: Union[int, str]=0) \
    -> Generator[Union[Tuple[List, List], Tuple[List, List, np.ndarray]]]:
    """Yields landmarks from blazepose from a video steam.

    Args:
        window_name (str, optional): Name of the window. Defaults to "blaze".
        file_name (str, optional): Prefix of file to write recording to. Defaults to "last_recording".
        model_complexity (int, optional): Blazepose model compelxity where 1 is the smallest
        model and 3 is the largest. Defaults to 1.
        return_image (bool, optional): If to return the cv2 image. If this is True
        than you must call imshow on the image to display the frame manually. Defaults to False.
        stream (Union[int, str]): Video stream to capture. Defaults to 0 which
        is the first webcame on your comptuer. Give filepath to capture from file.

    Yields:
        Generator[Union[Tuple[List, List], Tuple[List, List, np.ndarray]]]: landmarks, wlandmarks, 
        and optionally the image.
    """

    output_extension = '.csv'
    landmark_file_name = f"{file_name}_landmarks{output_extension}"
    wlandmark_file_name = f"{file_name}_wlandmarks{output_extension}"
    landmark_file = open(landmark_file_name, 'w')
    wlandmark_file = open(wlandmark_file_name, 'w')

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(stream)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=model_complexity
    ) as pose:
        cv2.imshow(window_name, np.zeros([2,2]))
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Video file ended..")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks is None:
                continue

            landmark, wlandmark = unpack_frame(results)
            write_frame(landmark, wlandmark, landmark_file, wlandmark_file)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # utilize drawing utilities for a pretty picture
            # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            image = cv2.flip(image, 1) 
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            if return_image:
                yield landmark, wlandmark, image
            else:
                cv2.imshow(window_name, image)
                yield landmark, wlandmark

    landmark_file.close()
    wlandmark_file.close()
    cap.release()
    cv2.destroyWindow(window_name)


def convert_landmarks_to_protobuf(data):
    assert isinstance(data, np.ndarray), "Data must be a numpy array"

    if len(data.shape) == 1:
        data = np.array([data])
    format_str = ""
    for frame in data:
        for point in frame.reshape([-1, 4]):
            landmark = 'landmark {x: ' + str(point[0]) + ' y: ' + \
                    str(point[1]) + ' z: ' + str(point[2]) + ' visibility: ' + str(point[3]) + '} '
            format_str += landmark

    landmark_list = text_format.Parse(format_str, landmark_pb2.NormalizedLandmarkList())
    return landmark_list


def vizualize(landmarks, image=None, name="Pose", wait=False):
    if isinstance(landmarks[0], float):
        landmarks = convert_landmarks_to_protobuf(landmarks)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles

    if image is None:
        image = np.zeros((400,400,3), np.uint8)
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    cv2.imshow(name, image)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    if wait:
        cv2.waitKey(0)

    return image


def average_landmarks(landmarks):
    landmarks_avg = np.mean(landmarks, axis=0)
    # Sum lengths over all connections
    lengths = j.blaze.lengths_from_landmarks(landmarks[0])
    for idx in range(1, landmarks.shape[0]):
        curr = j.blaze.lengths_from_landmarks(landmarks[idx])
        for k1, d1 in lengths.items():
            if d1 is not None:
                for k2, d2 in d1.items():
                    d1[k2] += curr[k1][k2]

    # Averge lengths
    for k1, d1 in lengths.items():
        if d2 is not None:
            for k2, d2 in d1.items():
                d1[k2] /= landmarks.shape[0]
    
    return j.blaze.fix_pose_avg(landmarks_avg, lengths)


def fix_pose_avg(landmarks_avgs, lengths:Dict[str, Dict[str, int]]):
    graph = deepcopy(j.blaze.BLAZEPOSE_GRAPH)
    frame = j.blaze.frame_to_dict(landmarks_avgs)
    graph["root"] = ["left_hip", "right_hip"]

    lengths["root"] = {}
    lengths["root"]["left_hip"] = lengths["left_hip"]["right_hip"] / 2
    lengths["root"]["right_hip"] = lengths["left_hip"]["right_hip"] / 2

    new_frame = {k: np.copy(v) for k, v in frame.items()}
    fix_pose_avg_helper("left_shoulder", graph, frame, lengths, new_frame)
    fix_pose_avg_helper("right_shoulder", graph, frame, lengths, new_frame)
    fix_pose_avg_helper("left_hip", graph, frame, lengths, new_frame)
    fix_pose_avg_helper("right_hip", graph, frame, lengths, new_frame)

    return j.blaze.dict_to_frame(new_frame)


def fix_pose_avg_helper(root, graph, frame, lengths, new_frame):
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



if __name__ == "__main__":
    for landmark, wlandmark in capture():
        continue
