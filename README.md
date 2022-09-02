# <img src="assets/CARRSQ_LOGO.jfif" width=30> Semantic Gaze Tracking

<img src="assets/GIFSEGMENTATION.gif" align="right" width=400 height=250>

This repository contains an explaination and demonstration of a proof of concept we created to merge signals from the eye tracker and the sensors seeing the vehicle environment.
The data used is provided by :
- A camera, on which we apply a semantic segmentation specific to road objects and environment
- A gaze tracker, that provides information about the head position and the gaze direction
- An advanced lidar system (here IBEO), that provides position and size of bounding boxes of the surrounding objects
- Another lidar (here Velodyne), that provides a dense point cloud of the surroundings

The image segmentation used is inspired by the [CSAILVision Github](https://github.com/CSAILVision/semantic-segmentation-pytorch).  
This project was conducted during a 5 months intership in the Centre of Accident Research and Road Safety ([CARRS-Q](https://research.qut.edu.au/carrsq/)) of the Queensland University of Technology (Australia).

## Context :

The problem
-	Driver’s distraction
-	Level 2 of vehicle autonomy : the driver must maintain their vigilance on the road
-	Level 3 : the driver is the fallback solution if the automation understands that it will fail to operate in a near future
-	Level 4 : the automation needs to understand the readiness of the driver if they decide to take over
- Current driver monitoring systems do not analyse the driver visual attention with respect to the environment
Proposed solution
-	Analyse the road and the driver’s focus to infer the danger rating of the situation

Targets
-	Identify the position of the objects looked at by the driver
-	Identify the class of the objects looked at
Problems currently
-	The gaze tracker gives information about the direction of the gaze but the information about the distance of the focus point is not precise enough (intersection of the 2 gaze lines)
-	In order to compute the distance of the objects, we need a 3D representation of the environment
-	Solution 1 : Stereo depth map images (we need a stereo camera system or a machine learning solution to recreate this image)
-	Solution 2 : Using a lidar (IBEO / Lidar)
-	Finding the class of the objects


## Processing Pipeline :

![image](https://user-images.githubusercontent.com/67725628/188056527-cb03d8a5-7de5-491e-8418-4c321421b3d7.png)
