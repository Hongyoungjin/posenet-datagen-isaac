# Generating data for pose detecting neural network 


## Introduction
This code basically generates training data for the neural network.
The neural network is for estimating the pose (position and orientation) of the seen object through the depth camera.
To note, this network only estimates  the pose of the certain object. For this case, we have tested the [EGAD!](https://github.com/dougsm/egad) evaluation datset. The obj and urdf files are included in the repo.
The deph camaera outputs the point cloud, the network estimates the position and orientation of the object using that point cloud.

For generating the training data for the neural network, we used [isaac gym](https://developer.nvidia.com/isaac-gym) simulator to place the object and get depth image, segmentation mask, and object's pose.

Object pose data are used as the label when training the network, and depth images and segmentation masks are used to generate point cloud to train the network.


## Prerequisites

### Install Isaac Gym

Go to [this link](https://developer.nvidia.com/isaac-gym) and donwload isaac gym

### Install required libraries

```bash
pip install -r requirements.txt
```
## Run the file
### Things for configurate

As noted in the introduction, the network is only for estimating pose of a single type of object. 
Therefore, before running the data generating code, you need to specify which object you are interested in.

We have added EGAD! evaluation objects in 'src' folder, so you can choose the object, and add the name on the config.yaml file in 'cfg' folder. In that file, add that name next to 'target_object' section.

One of the advantages of using isaac gym is that you can run multiple environments simultaneously. For example, if you add the number of environments that you want to run at once next to the 'num_envs' section in config.yaml.
It is recommended to apply 100 or 10.

### Run file command
If you want to run the code without opening the isaac gym window, add "--headless" option.
If you wan to save the output data, add "--save_results" option.

```bash
python3 main.py --headless --save_results
```
