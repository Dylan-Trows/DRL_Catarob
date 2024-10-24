#!/bin/bash

# Get current date and time
DATETIME=$(date +"%Y%m%d_%H%M%S")

# Prompt for scenario type
echo "Enter scenario type code (e.g., PT, HE, MR):"
read SCENARIO_TYPE

# Prompt for start position
echo "Enter start position code (e.g., A, B, C):"
read START_POSITION

# Prompt for attempt number
echo "Enter attempt number (e.g., 01, 02):"
read ATTEMPT_NUMBER

# Construct filename
FILENAME="${DATETIME}_${SCENARIO_TYPE}_${START_POSITION}_${ATTEMPT_NUMBER}"

# Start recording
ros2 bag record -o $FILENAME \
    /sensor/emlid_gps_fix \
    /sensors/mag_heading \
    /platform/hull/PORT_status \
    /platform/hull/STBD_status