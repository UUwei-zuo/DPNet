# DPNet: Doppler LiDAR Motion Planning for Highly-Dynamic Environments

<a href="https://arxiv.org/pdf/2512.00375"><img src='https://img.shields.io/badge/PDF-Arxiv-brightgreen' alt='PDF'></a>
<a href="https://youtu.be/0-fzaFXYKvg"><img src='https://img.shields.io/badge/Video-Youtube-blue' alt='youtube'></a>

This is the project page of the paper: **[DPNet: Doppler LiDAR Motion Planning for Highly-Dynamic Environments](https://arxiv.org/abs/2512.00375)**.

## Introduction

By leveraging **Doppler LiDAR** to obtain **real-time environmental velocity infomation**, we established a model-based learning method (Doppler Kalman Neural Network) that **tracks and predicts highly-dynamic obstacle motions**, and a model predictive control tuning method (Doppler-Tuned MPC) that **enables collision-free runtime controller tuning**. These two innovations are integrated into a unified solution, **DPNet**.

*DPNet is among the first pioneers to integrate the novel Doppler LiDAR sensors into closed-loop motion planning.*

**Watch the video introduction on YouTube:**

[![Watch the video](https://img.youtube.com/vi/0-fzaFXYKvg/maxresdefault.jpg)](https://youtu.be/0-fzaFXYKvg)


Full code will be released upon paper acceptance.
In the meantime, we provide [ros-bridge-DopplerLiDAR](https://github.com/UUwei-zuo/ros-bridge-DopplerLiDAR), a ROS bridge that links CARLA Doppler LiDAR support to ROS.

