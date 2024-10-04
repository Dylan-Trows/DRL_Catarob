"""
Process and align offline datasets from ROS bag files.
This script reads ROS bag files, processes GPS, heading, and hull data,
and aligns them into a format suitable for machine learning tasks.
"""
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import StorageOptions, ConverterOptions
import numpy as np
from geometry_msgs.msg import Twist
import math
from scipy.interpolate import interp1d
from haversine import haversine
from numba import jit
import sys
import os
import h5py
import argparse
sys.path.append('/home/dylan_trows/Documents')
from Reward_Calculator import RewardCalculator
import Offline_dataset_utility 

# Earth's radius in meters
R = 6371000

jit(nopython = True)
def magnetic_to_true_heading(magnetic_heading_data, declination):
    true_heading_data = [(x, (y + declination + 360)%360) for x, y in magnetic_heading_data]
    return true_heading_data

jit(nopython=True)
def calculate_velocity(x1, y1, x2, y2, time_step):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance / time_step

def process_bag_file(bag_path):
    """
    Process a ROS bag file and extract GPS, heading, and hull data.
    
    Args:
    bag_path: Path to the ROS bag file
    
    Returns:
    gps_data, heading_data, hull_data: Extracted data from the bag file
    """
    storage_options = StorageOptions(
        uri=bag_path,
        storage_id='sqlite3'
    )
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

    gps_data = []
    heading_data = []
    hull_data = {'PORT': [], 'STBD': []}

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic not in type_map:
            print(f"Warning: Unknown topic {topic}")
            continue

        try:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/sensor/emlid_gps_fix':
                gps_data.append((t, msg.latitude, msg.longitude, msg.altitude))
            elif topic == '/sensors/mag_heading':
                heading_data.append((t, msg.data))
            elif topic == '/platform/hull/PORT_status' or topic == '/platform/hull/STBD_status':
                hull_type = 'PORT' if 'PORT' in topic else 'STBD'
                hull_data[hull_type].append((t, msg.actual_pwm_rc))
        except Exception as e:
            print(f"Error processing message on topic {topic}: {str(e)}")

    return gps_data, heading_data, hull_data

jit(nopython=True)
def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the bearing between two GPS coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - \
        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    theta = math.atan2(y, x)
    #print("Theta = ", theta)                                                                # print statement for testing
    return (math.degrees(theta) + 360) % 360

jit(nopython=True)
def calculate_heading_error(current_heading: float, desired_heading: float) -> float:
    """Calculate the smallest angle between current and desired heading."""
    error = desired_heading - current_heading
    #print("Error", error)
    return (error + 180) % 360 - 180

jit(nopython=True)
def pwm_to_cmd_vel(port_pwm, stbd_pwm):
    """ 
    Convert PWM to linear and angular velocity
    according to the prop_driver.py code
    """
    max_pwm = 255  
    min_pwm = 0  
    
    stbd_vel = 2 + (4/255)*(stbd_pwm-255)
    port_vel = 2 + (4/255)*(port_pwm-255)

    A = np.array([[1,-0.42], [1, 0.42]])            #fwd_cmd - 0.42*rot_vel = stbd_vel
    b = np.array([stbd_vel, port_vel])              #fwd_cmd + 0.42*rot_vel = port_vel

    solution = np.linalg.solve(A,b)
    fwd_cmd = solution[0]
    rot_vel = solution[1]
        
    rot_cmd = rot_vel * 0.42
   
    return fwd_cmd, rot_cmd
def unwrap_angles(angles):
    return np.unwrap(np.deg2rad(angles))

def wrap_to_360(angles):
    return angles % 360

def interpolate_data(data, target_freq=4, headings = False):
    times = np.array([t for t, *_ in data])
    values = np.array([list(d[1:]) for d in data])
    
    start_time = times[0]
    end_time = times[-1]
    new_times = np.arange(start_time, end_time, 1e9/target_freq)  # 1e9 nanoseconds per second
    
    interpolated_values = []

    for i in range(values.shape[1]):
        column_values = values[:, i]
        if headings:
            print("Column values ", column_values)
            column_values = unwrap_angles(column_values)
            print("Unwrapped values (rad):", column_values)
            print("Unwrapped values (deg):", np.rad2deg(column_values))
        
        interp_func = interp1d(times, column_values, kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated_column = interp_func(new_times)
        
        if headings:
            interpolated_column = wrap_to_360(np.rad2deg(interpolated_column))
        
        interpolated_values.append(interpolated_column)
    
    return list(zip(new_times, *interpolated_values))

def align_and_convert_data(gps_data, heading_data, hull_data, final_waypoint, failure=False):
    """
    Align and convert data from different sources into a unified format.
    
    Args:
    gps_data: GPS data from the bag file
    heading_data: Heading data from the bag file
    hull_data: Hull data from the bag file
    final_waypoint: Final destination waypoint
    
    Returns:
    aligned_data: List of aligned and processed data points
    ref_point: Reference point (lat, lon) for coordinate conversions (start point)
    """
    reward_calculator = RewardCalculator()

    # Upsample all data to 4 Hz
    gps_data_upsampled = interpolate_data(gps_data,10)
    heading_data_upsampled = interpolate_data(heading_data, 10, True)
    hull_data_port_upsampled = interpolate_data(hull_data['PORT'],10)
    hull_data_stbd_upsampled = interpolate_data(hull_data['STBD'],10)

    # Get the shortest length among all upsampled data
    min_length = min(len(gps_data_upsampled), len(heading_data_upsampled), 
                     len(hull_data_port_upsampled), len(hull_data_stbd_upsampled))
    
    # Trim all data to the minimum length
    gps_data_upsampled = gps_data_upsampled[:min_length]
    heading_data_upsampled = heading_data_upsampled[:min_length]
    hull_data_port_upsampled = hull_data_port_upsampled[:min_length]
    hull_data_stbd_upsampled = hull_data_stbd_upsampled[:min_length]

    # Set the waypoint as the last GPS point
    if not failure:
        final_waypoint = gps_data_upsampled[-1][1:3]  # lat, lon of the last point
    
    # Get reference point (first GPS point)
    ref_lat, ref_lon = gps_data[0][1], gps_data[0][2]
    print("Position of Ref GPS point °: ", ref_lat," ", ref_lon)
    print("Position of upsampled waypoint °: ", final_waypoint[0], final_waypoint[1])
    print("Position of Upsampled waypoint :",Offline_dataset_utility.latlon_to_xy(final_waypoint[0],final_waypoint[1], ref_lat, ref_lon))
    print("Distance to Upsampled waypoint: ",  haversine((ref_lat, ref_lon), (final_waypoint[0], final_waypoint[1]), unit='m') )

    waypoint_x, waypoint_y = Offline_dataset_utility.latlon_to_xy(final_waypoint[0], final_waypoint[1], ref_lat, ref_lon)

    aligned_data = []
    prev_x, prev_y = Offline_dataset_utility.latlon_to_xy(gps_data_upsampled[0][1], gps_data_upsampled[0][2], ref_lat, ref_lon)

    for i in range(min_length):  #  to ensure we have a next state for each current state
        t, lat, lon, alt = gps_data_upsampled[i]
        heading = heading_data_upsampled[i][1]
        port_pwm = hull_data_port_upsampled[i][1]
        stbd_pwm = hull_data_stbd_upsampled[i][1]

        # Convert lat/lon to x/y
        x, y = Offline_dataset_utility.latlon_to_xy(lat, lon, ref_lat, ref_lon)
    
        # Calculate heading error      
        desired_heading = calculate_bearing(lat, lon, final_waypoint[0], final_waypoint[1])
        heading_error = calculate_heading_error(heading, desired_heading)
        
        # Convert PWM to cmd_vel
        linear_x, angular_z = pwm_to_cmd_vel(port_pwm, stbd_pwm)

        distance_to_waypoint = math.sqrt((x - waypoint_x)**2 + (y - waypoint_y)**2)
        arc_length = distance_to_waypoint * math.radians(abs(heading_error))
        velocity = calculate_velocity(prev_x, prev_y, x, y, 0.25)  # 0.25 seconds for 4 Hz
        #print("Velocity at time ",i,": ",round(velocity, 2))
        done = 1 if distance_to_waypoint <1.0 or i == min_length -1 else 0

        # Create state and action
        state = np.array([
            round(x, 2),               # X position (meters)
            round(y, 2),               # Y position (meters)
            round(heading, 1),         # Heading (degrees)
            round(heading_error, 1),   # Heading error (degrees)
            round(distance_to_waypoint, 2),  # Distance to waypoint (meters)
            round(arc_length, 2),
            round(velocity, 2),
            round(waypoint_x, 2),      # Waypoint X (meters)
            round(waypoint_y, 2),      # Waypoint Y (meters)
            int(done)                  # Done flag (0 or 1)
        ])

        action = np.array([round(linear_x, 2), round(angular_z, 2)])

        if i < min_length - 1:
            # Get next state
            next_t, next_lat, next_lon, next_alt = gps_data_upsampled[i+1]
            next_x, next_y = Offline_dataset_utility.latlon_to_xy(next_lat, next_lon, ref_lat, ref_lon)
            next_heading = heading_data_upsampled[i+1][1]
            next_desired_heading = calculate_bearing(next_lat, next_lon, final_waypoint[0], final_waypoint[1])
            next_heading_error = calculate_heading_error(next_heading, next_desired_heading)
            next_distance_to_waypoint = math.sqrt((x - waypoint_x)**2 + (y - waypoint_y)**2)
            next_arc_length = next_distance_to_waypoint * math.radians(abs(heading_error))
            next_velocity = calculate_velocity(x, y, next_x, next_y, 0.25)  # 0.25 seconds for 4 Hz
            #print("Next velocity at ",i,": ",round(next_velocity, 2))
            next_state = np.array([
                round(next_x, 2),
                round(next_y, 2),
                round(next_heading, 1), 
                round(next_heading_error, 1),
                round(next_distance_to_waypoint, 2),
                round(next_arc_length, 2),
                round(next_velocity, 2),
                round(waypoint_x, 2), 
                round(waypoint_y, 2),
                int(done)
            ])
        else:
            next_state = state  # For the last state, use current state as next state
        
        # Calculate reward
        reward = reward_calculator.calculate_reward(state, action, (waypoint_x, waypoint_y))
        
        aligned_data.append((state, action, round(reward, 4), next_state, done))

        if done:
            break

        prev_x, prev_y = x, y
    
    print("Total Reward: ", reward_calculator.total_reward)

    return aligned_data, (ref_lat, ref_lon)

def compare_distance_calculations(lat1, lon1, lat2, lon2, ref_lat, ref_lon):
    """
    used to verify distrance calculations between the Haversine formula and the XY distances after coordinate system change
    """
    # Haversine distance
    haversine_dist = haversine((lat1, lon1), (lat2, lon2), unit='m')

    # X-Y distance
    x1, y1 = Offline_dataset_utility.latlon_to_xy(lat1, lon1, ref_lat, ref_lon)
    x2, y2 = Offline_dataset_utility.latlon_to_xy(lat2, lon2, ref_lat, ref_lon)
    xy_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    print(f"Haversine distance: {haversine_dist:.2f} m")
    print(f"X-Y distance: {xy_dist:.2f} m")
    print(f"Difference: {abs(haversine_dist - xy_dist):.2f} m")

def process_single_trajectory(bag_path, trajectory_type, name, output_dir):
    
    GPS_middle_of_vlei = (-34.0898934162943, 18.466624778750354)        # using position of upsampled waypoint Perfect 01
    GPS_shore_of_vlei_start = (-34.09029552333333, 18.46672054)         # using ref position of gps data for perfect 01
    #GPS_midlle_location_b = (-34.08994286, 18.46656999)                 # original waypoint in trajectory 36 location_b 
    #GPS_Shore_location_b = (-34.090108566666665, 18.467099085)          # using ref position of trajectory 36 loction_b
    
    gps_data, heading_data, hull_data = process_bag_file(bag_path)

    # Get reference point (first GPS point)
    ref_lat, ref_lon = gps_data[0][1], gps_data[0][2]

    # Adjust heading data for magnetic declination
    heading_data = magnetic_to_true_heading(heading_data, -26.6)

    # Get the final GPS position as the destination waypoint
    final_waypoint = gps_data[-1][1:]  # lat, lon, alt of the last GPS point
    print("Position of Original waypoint in °: ", final_waypoint[0], final_waypoint[1])
    print("Position of Original waypoint :",Offline_dataset_utility.latlon_to_xy(final_waypoint[0],final_waypoint[1], ref_lat, ref_lon))
    print("Distance to Original waypoint: ", haversine((ref_lat, ref_lon), (final_waypoint[0], final_waypoint[1]), unit='m') )

    # TODO
    # Align and convert data
    processed_data, ref_point = align_and_convert_data(gps_data, heading_data, hull_data, final_waypoint)
    
    # for failure trajectories
    processed_data, ref_point = align_and_convert_data(gps_data, heading_data, hull_data, GPS_middle_of_vlei, True)
    
    Offline_dataset_utility.print_sample_data(processed_data)
    # Plot actual trajectory used

    # Plot actual trajectory used
    Offline_dataset_utility.plot_trajectory(processed_data, ref_point[0], ref_point[1])

    trajectory_name = "trajectory_" + name 
    # Save processed data
    file = f'{trajectory_name}.h5'
    filepath = os.path.join(output_dir, file)
    with h5py.File(filepath, 'w') as hf:
        for i, step in enumerate(processed_data):
            step_group = hf.create_group(f'step_{i}')
            step_group.create_dataset('state', data=step[0])
            step_group.create_dataset('action', data=step[1])
            step_group.create_dataset('reward', data=step[2])
            step_group.create_dataset('next_state', data=step[3])
            step_group.create_dataset('done', data=step[4])
    
    print(f"Saved trajectory to {filepath}")
    print(f"Processed {len(processed_data)} data points.")

def main(args):

    """# GPS_middle_of_vlei = (-34.0898934162943, 18.466624778750354)        # using position of upsampled waypoint Perfect 01
    # GPS_shore_of_vlei_start = (-34.09029552333333, 18.46672054)         # using ref position of gps data for perfect 01
    # #GPS_midlle_location_b = (-34.08994286, 18.46656999)                 # original waypoint in trajectory 36 location_b 
    # #GPS_Shore_location_b = (-34.090108566666665, 18.467099085)          # using ref position of trajectory 36 loction_b

    # # Process the bag file
    # bag_path = args.bag_folder
    # trajectory_type = args.trajectory_type
    # trajectory_name = 'trajectory_'+args.trajectory_name

    # gps_data, heading_data, hull_data = process_bag_file(bag_path)

    # # Get reference point (first GPS point)
    # ref_lat, ref_lon = gps_data[0][1], gps_data[0][2]

    # # Adjust heading data for magnetic declination
    # heading_data = magnetic_to_true_heading(heading_data, -26.6)

    # Interpolate the data
    # gps_data_upsampled = interpolate_data(gps_data)
    # for i in range(1, len(gps_data_upsampled), 10):  # Check every 100th point
    #     lat1, lon1 = gps_data_upsampled[0][1], gps_data_upsampled[0][2]
    #     lat2, lon2 = gps_data_upsampled[i][1], gps_data_upsampled[i][2]
    #     print(f"\nComparing point 0 to point {i}:")
    #     compare_distance_calculations(lat1, lon1, lat2, lon2, ref_lat, ref_lon)
    
    # heading_data_upsampled = interpolate_data(heading_data,4,True)
    # heading_data_test = interpolate_data(heading_data,4)
    # print("Differences between wrap Heading: ")
    # for i in range (len(heading_data_test)):
    #     print(heading_data_upsampled[i][1]," : ", heading_data_test[i][1])
    # hull_data_port_upsampled = interpolate_data(hull_data['PORT'])
    # hull_data_stbd_upsampled = interpolate_data(hull_data['STBD'])

    # Plot comparisons
    # Plot XY comparisons
    
    Offline_dataset_utility.plot_xy_interpolation(gps_data, gps_data_upsampled, ref_lat, ref_lon, trajectory_type+'/'+trajectory_name)
    Offline_dataset_utility.plot_gps_interpolation(gps_data, gps_data_upsampled, ref_lat, ref_lon, trajectory_type+'/'+trajectory_name)
    Offline_dataset_utility.plot_interpolated_vs_original(heading_data, heading_data_upsampled, 'Heading', trajectory_type+'/'+trajectory_name)
    Offline_dataset_utility.plot_interpolated_vs_original(hull_data['PORT'], hull_data_port_upsampled, 'Port PWM',trajectory_type+'/'+trajectory_name)
    Offline_dataset_utility.plot_interpolated_vs_original(hull_data['STBD'], hull_data_stbd_upsampled, 'Starboard PWM',trajectory_type+'/'+trajectory_name)
    
    # # Get the final GPS position as the destination waypoint
    # final_waypoint = gps_data[-1][1:]  # lat, lon, alt of the last GPS point
    # print("Position of Original waypoint in °: ", final_waypoint[0], final_waypoint[1])
    # print("Position of Original waypoint :",Offline_dataset_utility.latlon_to_xy(final_waypoint[0],final_waypoint[1], ref_lat, ref_lon))
    # print("Distance to Original waypoint: ", haversine((ref_lat, ref_lon), (final_waypoint[0], final_waypoint[1]), unit='m') )

    # TODO
    # Align and convert data
    # processed_data, ref_point = align_and_convert_data(gps_data, heading_data, hull_data, final_waypoint)
    
    # # for failure trajectories
    # #processed_data, ref_point = align_and_convert_data(gps_data, heading_data, hull_data, GPS_middle_of_vlei, True)

    # Offline_dataset_utility.print_sample_data(processed_data)
    # # Plot actual trajectory used
    # Offline_dataset_utility.plot_trajectory(processed_data, ref_point[0], ref_point[1])
    # #Offline_dataset_utility.plot_trajectory(processed_data, ref_point[0], ref_point[1],trajectory_type+'/'+trajectory_name)
    # # Save processed data
    # file = trajectory_name+'.h5'
    # filepath = os.path.join("/home/dylan_trows/Documents/dataset_auto_test", file)
    # with h5py.File(filepath, 'w') as hf:
    #     for i, step in enumerate(processed_data):
    #             step_group = hf.create_group(f'step_{i}')
    #             step_group.create_dataset('state', data=step[0])
    #             step_group.create_dataset('action', data=step[1])
    #             step_group.create_dataset('reward', data=step[2])
    #             step_group.create_dataset('next_state', data=step[3])
    #             step_group.create_dataset('done', data=step[4])
        
    # print(f"Saved trajectory to {filepath}")
    # print(f"Processed {len(processed_data)} data points.")"""

    
    root_dir = args.root_folder
    output_dir = args.output_folder

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # trajectory_number = 
    # process_single_trajectory(bag_path, "perfect", str(trajectory_number), output_dir)


    # for trajectory_type in os.listdir(root_dir):
    #     type_dir = os.path.join(root_dir, trajectory_type)
    #     if os.path.isdir(type_dir):
    #         for trajectory_name in os.listdir(type_dir):
    #             trajectory_dir = os.path.join(type_dir, trajectory_name)
    #             if os.path.isdir(trajectory_dir):
    #                 # Find the bag file in this directory
    #                 bag_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.db3')]
    #                 if bag_files:
    #                     bag_path = os.path.join(trajectory_dir, bag_files[0])
    #                     print(f"Processing trajectory: {trajectory_type}/{trajectory_name}")
    #                     process_single_trajectory(bag_path, trajectory_type, trajectory_name, output_dir)
    #                 else:
    #                     print(f"No bag file found in {trajectory_dir}")

    trajectory_number = 36
    for bag_folder in sorted(os.listdir(root_dir)):
        bag_path = os.path.join(root_dir, bag_folder)
        if os.path.isdir(bag_path):
            print(f"Processing trajectory {trajectory_number}: {bag_folder}")
            process_single_trajectory(bag_path, "perfect", str(trajectory_number), output_dir)
            trajectory_number += 1

    print(f"Total trajectories processed: {trajectory_number}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ROS bag file and generate trajectory data.")
    """parser.add_argument("bag_folder", help="Path to the ROS bag folder")
    parser.add_argument("trajectory_type", help="Type of the trajectory")
    parser.add_argument("trajectory_name", help="Name of the trajectory")"""

    parser.add_argument("root_folder", help="Path to the root folder containing trajectory type folders")
    parser.add_argument("output_folder", help="Path to the output folder for processed .h5 files")
    
    args = parser.parse_args()

    main(args)