# WaymoProject
# First Part: Waymo Motion Dataset Visualizer from Tutorial(DID NOT WRITE CODE FOR THIS)

A lightweight Python tool for visualizing agent trajectories, roadgraphs, and traffic light states from the **Waymo Open Motion Dataset** tutorial.  
This script decodes a `.tfrecord` file and generates a smooth animation of all agents (vehicles, pedestrians, cyclists) over time.

---

## Example Animation

Below is a sample of the generated trajectory animation:

![Waymo Motion Visualization](./tutorial_work/anim.gif)

---

## Features
- Parses **Waymo Motion Dataset** TFRecord files directly using TensorFlow.
- Visualizes:
  - Past, current, and future agent trajectories.
  - Roadgraphs and traffic lights.
- Exports animations as `.mp4`, `.gif`, or `.html`.
- Compatible with **Docker**, **VS Code**, and **Jupyter Lab** environments.

---

## Citations
https://github.com/waymo-research/waymo-open-dataset

@InProceedings{Ettinger_2021_ICCV, author={Ettinger, Scott and Cheng, Shuyang and Caine, Benjamin and Liu, Chenxi and Zhao, Hang and Pradhan, Sabeek and Chai, Yuning and Sapp, Ben and Qi, Charles R. and Zhou, Yin and Yang, Zoey and Chouard, Aur'elien and Sun, Pei and Ngiam, Jiquan and Vasudevan, Vijay and McCauley, Alexander and Shlens, Jonathon and Anguelov, Dragomir}, title={Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset}, booktitle= Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, month={October}, year={2021}, pages={9710-9719} }

@InProceedings{Kan_2024_icra, author={Chen, Kan and Ge, Runzhou and Qiu, Hang and Ai-Rfou, Rami and Qi, Charles R. and Zhou, Xuanyu and Yang, Zoey and Ettinger, Scott and Sun, Pei and Leng, Zhaoqi and Mustafa, Mustafa and Bogun, Ivan and Wang, Weiyue and Tan, Mingxing and Anguelov, Dragomir}, title={WOMD-LiDAR: Raw Sensor Dataset Benchmark for Motion Forecasting}, month={May}, booktitle= Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)}, year={2024} }
