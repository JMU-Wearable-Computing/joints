import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joints as j
from skillest.task_decomp.blaze_pose_seg import get_all_2d_angles, segment
from skillest.task_decomp.segmentation import Segmentation


jj = pd.read_csv("tests/jumping_jack_blaze.csv", index_col="timestamp").values[25:]
data, angle_dict, idx_dict = get_all_2d_angles(jj)


if __name__ == "__main__":
    # seg = Segmentation(k=2)
    # points, _ = seg.segment(data, return_properties=True)

    # seg.plot_segmentation(np.tile(data, [1, 1]), angle_dict.keys(), True)
    # plt.show()
    frame = jj[56]
    jc = j.JointCollection(j.blaze.frame_to_dict(frame), j.blaze.BLAZEPOSE_CONNECTIONS)
    s = j.blaze.BLAZE_LEFT_ELBOW()
    s.set_joint_collection(jc)
    # print(jc["left_shoulder"])
    # print(jc["left_hip"])
    # print(jc["left_knee"])
    print( (np.array(list(s.angle().values())) * 180) / np.pi)
    # print((180 * np.arccos(jc["left_hip", "left_shoulder", "left_elbow"].as_quat()[-1]) * 2) / np.pi)
    # print((180 * np.arccos(jc["right_hip", "right_shoulder", "right_elbow"].as_quat()[-1]) * 2) / np.pi)
