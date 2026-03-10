# py_stereo_vo

A lightweight **Stereo Visual Odometry / SLAM system implemented in Python** for research and experimentation with classical geometric vision methods.

The system processes stereo image sequences and estimates camera motion while building a sparse 3D landmark map. The architecture is modular so that additional components such as loop closure, global bundle adjustment, or deep learning models can be integrated later.

The current implementation supports the **KITTI Odometry Dataset** and provides **real-time visualization of the estimated trajectory and landmarks**.

---

## Demo

A short demonstration of the system running on a KITTI sequence is shown below.

[demo video](https://github.com/Jacob66Y/py_stereo_vo/blob/local_ba/demo_trim.mp4)

---

# Features

- Stereo visual odometry pipeline
- Real-time trajectory visualization
- Sparse 3D landmark map generation
- Feature detection using **GFTT**
- Feature tracking using **Lucas–Kanade optical flow**
- Stereo matching for depth estimation
- Camera motion estimation using **PnP + RANSAC**
- 3D landmark triangulation
- Local **Bundle Adjustment** backend optimization

---

# System Pipeline

The current implementation follows a classical **feature-based visual SLAM pipeline**:
Stereo Images
│
▼
Feature Detection (GFTT)
│
▼
Feature Tracking (Lucas–Kanade Optical Flow)
│
▼
Stereo Matching
│
▼
Pose Estimation (PnP + RANSAC)
│
▼
Triangulation
│
▼
Map Update
│
▼
Backend Optimization (Local Bundle Adjustment)

---

# Backend Optimization

The backend performs **local bundle adjustment** to jointly optimize:

- camera poses
- 3D landmark positions

The optimization minimizes the **reprojection error**:

\[
\min_{T_i,X_j}
\sum_{i,j}
\left\|
z_{ij} - \pi(T_i,X_j)
\right\|^2
\]

Where

- \(T_i\) : camera pose  
- \(X_j\) : landmark position  
- \(z_{ij}\) : observed feature point  

The backend runs in a **separate thread** to avoid blocking the real-time frontend tracking.

---

# Dataset

The system currently supports the **KITTI Odometry Dataset**.

Download from:

http://www.cvlibs.net/datasets/kitti/eval_odometry.php

Expected folder structure:

dataset/
├── sequences/
│ ├── 00/
│ │ ├── image_0
│ │ ├── image_1
│ │ └── times.txt
│
└── poses/
└── 00.txt


---

# Installation

Clone the repository

git clone https://github.com/USER_NAME/py_stereo_vo.git

cd py_stereo_vo

Create a virtual environment

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

---

# Running the System

Run the visual odometry pipeline:

python run_vo.py --config configs/kitti_gray_test.yaml

The system will

- load the dataset sequence
- run stereo visual odometry
- visualize the trajectory in real time
- display the estimated map landmarks

---

# Current Limitations

- No loop closure
- No pose graph optimization
- Limited landmark management
- Backend bundle adjustment not GPU accelerated

---

# Future Work

Planned improvements include

- loop closure detection
- pose graph optimization
- global bundle adjustment
- GPU accelerated optimization
- integration of deep learning models for
  - depth estimation
  - feature extraction
  - motion prediction
- support for monocular visual SLAM

---

# Author

Jacob Yang  
