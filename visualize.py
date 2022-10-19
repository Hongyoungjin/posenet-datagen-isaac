import matplotlib.pyplot as plt
import numpy as np
import os
dir = os.path.join(os.getcwd(),"src/E2/data")

for file_idx in range(0,100):
    pose_name = ("pose_%05d.npy")%(file_idx)
    mask_name = ("mask_%05d.npy")%(file_idx)
    depth_name = ("image_%05d.npy")%(file_idx)

    pose = np.load(dir + "/" + pose_name, allow_pickle=True)
    mask = np.load(dir + "/" + mask_name, allow_pickle=True)
    depth = np.load(dir + "/" + depth_name, allow_pickle=True)

    print(pose.shape)
    print(mask.shape)
    print(depth.shape)
    # print(pose.p)
    plt.figure()
    plt.imshow(depth)
    plt.show()