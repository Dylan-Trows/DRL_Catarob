# DRL_Catarob
# Catarob Deep Reinforcement Learning Controller

This repository contains the implementation of a Deep Reinforcement Learning (DRL) controller for autonomous waypoint navigation of the Catarob Unmanned Surface Vehicle (USV).

## Overview

The system implements offline DRL algorithms for USV waypoint navigation, trained on real-world data collected from manual control sessions. The controller is designed to handle the challenges of marine environments while maintaining efficient trajectory tracking.

The system consists of two main implementations:
- Online Training Version: Full DRL training capability with experience replay VRX simulator
- Offline Deployment Version: Optimized for real-world deployment using pre-trained models

### Key Components

1. **Controller Node**
   - Handles sensor data integration
   - Manages waypoint navigation
   - Processes GPS and heading data
   - Publishes state information

2. **DRL Agent Node**
   - Implements TD7 algorithm
   - Manages model inference
   - Generates control actions
   - Interfaces with vehicle control system

3. **Waypoint Manager**
   - GPS coordinate processing
   - Distance and bearing calculations
   - Waypoint tracking and transitions
   - Coordinate system conversions

Key Features

Offline DRL implementation 
ROS 2-based control architecture
Real-world data collection and processing pipeline
Comprehensive training and evaluation framework
Integration with Catarob USV platform

## Requirements

- ROS 2 Humble
- Python 3.8+
- PyTorch
- NumPy
- CUDA-capable GPU (recommended for training)
- Additional ROS 2 dependencies:
  - sensor_msgs
  - geometry_msgs
  - std_msgs
  - custom message types (catarob_drl_interfaces)

## Installation

1. Clone the repository into your ROS 2 workspace:
```bash
cd ~/ros2_ws/src
git clone [https://github.com/Dylan-Trows/DRL_Catarob]
cd ros2_ws
colcon build
source install/setup.bash

