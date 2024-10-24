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

2. To run nodes (Offline controller and agent) :
ros2 run catarob_drl catarob_controller_experimental

3. in a different terminal:
cd ros2_ws
source install/setup.bash
ros2 run catarob_drl catarob_drl_agent

4. If on the catarob: 
ros2 launch ~/catarob_ws/launch/catarob.launch.py	(to launch the catarob_ros2 package)

## Online versions 
Install the VRX simulator: 
follow instructions in https://github.com/osrf/vrx/blob/main/README.md
This requires installation of the correct versions of ros and gazebo, this is detailed in their GitHub

Once the VRX simulator is setup:
1. ros2 run catarob_drl vrx_controller
 
2. In a different terminal:
ros2 run catarob_drl drl_agent

3. In a third terminal:
ros2 launch vrx_gz competition.launch.py world:=sydney_regatta 



## Using the catarob
To Record bag data simply run: (using the custom bash script in the GitHub)

chmod +x record_catarob.sh	(make it executable)
./record_catarob.sh

OR

```
ros2 bag record -o 20240920_113000_PT_A_01 /sensor/emlid_gps_fix /sensors/mag_heading /platform/hull/PORT_status /platform/hull/STBD_status
```

As for naming the bag files:

```
YYYYMMDD_HHMMSS_ScenarioType_StartPosition_AttemptNumber.bag
```

Where:
- YYYYMMDD: Date of recording
- HHMMSS: Time of recording
- ScenarioType: A short code for the type of scenario
- StartPosition: A code for the starting position
- AttemptNumber: Which attempt this is for the given scenario

Coding system for ScenarioType:

- PT: Perfect Trip
- HE: Heading Error start
- MR: Mistake and Recovery
- WV: Wavy/Noisy trip
- SL: Slow trip
- MF: Mistake (Failed trip)
- FS: Forward and Stop
- MB: Moving Backwards
- RP: Reversing Perfect
- RW: Reversing Wavy
- TI: Turning in place
- FC: Forward and Coast

- VS: Variable Speed
- ZZ: Zigzag path
- SH: Stationary Holding
- TT: Tight Turn
- MS: Maximum Speed

For StartPosition,  A, B, C for starting positions.

Examples:
```
20230620_143000_PT_A_01.bag  (Perfect Trip from position A, first attempt)
20230620_144500_HE_B_02.bag  (Heading Error start from position B, second attempt)
20230620_150000_MR_C_01.bag  (Mistake and Recovery from position C, first attempt)
```

Remote Desktop
To connect to the Catarob, at this stage, remote desktop is to be used. There are currently three methods to connect to the Catarob, all are detailed here. These instructions assume the Catarob is powered on.

Connect over Ethernet

Connect the ethernet cable to your computer and to the switch in the Catarob
Open Remote desktop on your laptop
Use 192.168.1.97 as the IP Address [This is the catarob PC]
Enter the Username and Password:
Username: catarob
Password: subseatech
Connect over WiFi [Direct Link]

Connect the ethernet cable to your computer and the POE box
Connect the POE box to the Ubiquity WiFi antenna
Connect to the “Catarob” wifi network
Open Remote desktop on your laptop
Use 192.168.1.97 as the IP Address [This is the catarob PC]
Enter the Username and Password:
Username: catarob
Password: subseatech

