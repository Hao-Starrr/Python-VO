# Python-VO

![image](./ref.gif)

This repository implements a visual odometry system using Python. The system processes video frames to estimate the motion of a camera and reconstructs the 3D structure of the scene. The implementation leverages OpenCV for computer vision tasks and numpy for numerical operations.

A minimum python version mono visual odometry.
Pipeline:
1. Frame Processing: Converts each frame to grayscale to simplify and expedite subsequent feature detection and matching steps.
2. Feature Extraction and Matching: Leverages ORB for efficient feature detection and FLANN with LSH for fast, accurate feature matching, optimizing the identification of corresponding points across frames.
3. Motion Estimation: Uses RANSAC for robust Essential Matrix estimation to derive precise rotation and translation matrices, filtering outliers and ensuring reliable motion estimation.
4. Pose Update: Updates camera pose using calculated motion vectors, generating precise projection matrices for depth estimation and 3D reconstruction.
5. Triangulation: Triangulates matching points between successive frames to reconstruct 3D scene geometry, utilizing the camera's intrinsic parameters and updated poses for spatial reconstruction.
• Further improvement can be on G2O pose optimization, database maintainance, and multi-threading for faster processing.


## Requirements
- Python 3.8+
- OpenCV 4.5.1
- NumPy 1.20.0
