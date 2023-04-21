# B31XP-MultidisciplinaryProject
Semantic Classification and Navigation with Teach-and-Repeat Inspection

This repository contains semantic map algorithm for underwater vtr inspection

this script works in conjunction with yolov5ros package provided at https://github.com/mats-robotics/yolov5_ros.git
and ORB_SLAM3 : https://github.com/UZ-SLAMLab/ORB_SLAM3.git
During teach phase
-replace groundtruth robot pose to ORB_SLAM estimated pose for objectdetector and Semanticmapmaker.
-launch ORB_SLAM3
-launch yolov5 detection
-launch map creator

During repeat phase 
-rename object_map.csv as M1.csv
-rename projection.csv as C1.csv
-launch ORB_SLAM3
-launch yolov5 detection
-run relocatization.py


UUV_sim package used for underwater simulation check https://field-robotics-lab.github.io/dave.doc/contents/installation/Installation/
made changes to Rexrove camera sensor plugins to include depth stereo camera.
-replace camera_snippet.xacro

Gazebo testing World
Additional models used in the gazebo world can be downloaded from :
https://drive.google.com/drive/folders/1AlWjQZcZUk3bDQ0q0uT47cm55On0YdI8?usp=share_link



