"""
Utility functions for processing and visualizing offline datasets.
This module provides various functions for coordinate conversions, 
plotting trajectories, and data visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import jit
import os

# Earth's radius in meters
R = 6371000
base_path = '/home/dylan_trows/Documents/trajectory_plots_10hz'

jit(nopython=True)
def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    """
    Convert latitude and longitude to x, y coordinates.
    
    Args:
    lat, lon: Coordinates to convert
    ref_lat, ref_lon: Reference coordinates (origin of the xy plane)
    
    Returns:
    x, y: Converted coordinates in meters
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    d_lat = lat_rad - ref_lat_rad
    d_lon = lon_rad - ref_lon_rad
    x = R * d_lon * math.cos(ref_lat_rad)
    y = R * d_lat
    return x, y

jit(nopython=True)
def xy_to_latlon(x, y, ref_lat, ref_lon):
    """
    Convert x, y coordinates back to latitude and longitude.
    
    Args:
    x, y: Coordinates in meters
    ref_lat, ref_lon: Reference coordinates (origin of the xy plane)
    
    Returns:
    lat, lon: Converted coordinates in degrees
    """
    ref_lat_rad = math.radians(ref_lat)
    lat = math.degrees(y / R + ref_lat_rad)
    lon = math.degrees(x / (R * math.cos(ref_lat_rad)) + math.radians(ref_lon))
    return lat, lon



def plot_trajectory(aligned_data, ref_lat, ref_lon, file_path = None):
    """
    Plot the trajectory and action data from aligned_data.
    """
    # Extract x, y coordinates and actions from aligned_data
    x, y = zip(*[(state[0], state[1]) for state, _, _, _ ,_ in aligned_data])
    linear_x, angular_z = zip(*[action for _, action, _, _,_ in aligned_data])

    # Convert x, y back to lat, lon for plotting
    lats, lons = zip(*[xy_to_latlon(x_i, y_i, ref_lat, ref_lon) for x_i, y_i in zip(x, y)])

    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Plot trajectory
    plt.plot(x, y, 'b-')
    plt.axis('equal')  # This ensures 1:1 aspect ratio
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    # Find the range of data
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    max_range = max(x_range, y_range)
    
    # Set consistent major tick spacing (e.g., every 10 meters)
    major_tick_spacing = 10
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(base=major_tick_spacing))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(base=major_tick_spacing))

    # Adjust limits to be slightly larger than data range
    plt.xlim(min(x) - 0.1*x_range, max(x) + 0.1*x_range)
    plt.ylim(min(y) - 0.1*max_range, max(y) + 0.1*max_range)
    plt.title('Interpolated GPS Trajectory (X/Y)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'Interpolated_GPS_trajectory_m.png'))
    plt.show()

    plt.figure(figsize=(12, 8))  # Rectangular figure
    plt.plot(lons, lats, 'b-')
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()

    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title('Interpolated GPS Trajectory (lat/lon)')
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'Interpolated_GPS_trajectory_°.png'))
    plt.show()

    # Create the plot
    plt.figure(figsize=(10, 10))

    # Plot actions
    plt.subplot(2, 1, 1)
    plt.plot(range(len(linear_x)), linear_x, 'r-')
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    major_tick_spacing = 10
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(base=major_tick_spacing))
    plt.title('Linear Velocity')
    plt.xlabel('Time Step')
    plt.ylabel('Linear X (m/s)')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(angular_z)), angular_z, 'g-')
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    major_tick_spacing = 10
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(base=major_tick_spacing))
    plt.title('Angular Velocity')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Z (rad/s)')
    plt.tight_layout()
    

    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'Interpolated_Action.png'))
    plt.show()
    

def print_sample_data(aligned_data, num_samples=400):
    print("\nSample of aligned data:")
    print("Time | State (Lat, Lon, Heading, heading error, dist, l arc, velocity, Way X, Way Y, done) | Action (Linear X, Angular Z) | Reward | Next State | done")
    for i, (state, action, reward, next_state, done) in enumerate(aligned_data[:num_samples]):
        print(f"{i}: {state} | {action} | {reward:.4f} | {next_state} | {done}")

def plot_interpolated_vs_original(original_data, interpolated_data, title, file_path=None):
    original_times = np.array([t for t, _ in original_data]) - original_data[0][0]
    original_values = np.array([v for _, v in original_data])
    
    interpolated_times = np.array([t for t, _ in interpolated_data]) - interpolated_data[0][0]
    interpolated_values = np.array([v for _, v in interpolated_data])
    
    plt.figure(figsize=(12, 6))
    plt.scatter(original_times / 1e9, original_values, label='Measured', alpha=0.5)
    if title == 'Heading':
        for i in range(len(interpolated_times)):
            if abs(interpolated_values[i] - interpolated_values[i-1]) < 45:
                plt.plot(interpolated_times[i-1:i+1] / 1e9, interpolated_values[i-1:i+1], 'r-', label='Interpolated' if i==0 else '', alpha=0.7)
    else:
        plt.plot(interpolated_times / 1e9, interpolated_values, 'r-', label='Interpolated', alpha=0.7)
    plt.title(f'{title} - Measured vs Interpolated')
    plt.xlabel('Time (s)')
    if title == 'Heading':
        plt.ylabel('Degrees (°)')
    elif title == 'Port PWM' or title == 'Starboard PWM':
        plt.ylabel('PWM Value')
    else:
        plt.ylabel('Value')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{title.lower().replace(" ", "_vs_interpolated")}.png'))
    plt.show()

def plot_gps_interpolation(original_gps, interpolated_gps, ref_lat, ref_lon, file_path = None):
    # Prepare original GPS data
    original_times = np.array([t for t, _, _, _ in original_gps]) - original_gps[0][0]
    original_lats = np.array([lat for _, lat, _, _ in original_gps])
    original_lons = np.array([lon for _, _, lon, _ in original_gps])

    # Prepare interpolated GPS data
    interpolated_times = np.array([t for t, _, _, _ in interpolated_gps]) - interpolated_gps[0][0]
    interpolated_lats = np.array([lat for _, lat, _, _ in interpolated_gps])
    interpolated_lons = np.array([lon for _, _, lon, _ in interpolated_gps])

    # Plot Latitude
    plt.figure(figsize=(12, 6))
    plt.scatter(original_times / 1e9, original_lats, label='Measured', alpha=0.5)
    plt.plot(interpolated_times / 1e9, interpolated_lats, 'r-', label='Interpolated', alpha=0.7)
    plt.title('Latitude - Measured vs Interpolated')
    plt.xlabel('Time (s)')
    plt.ylabel('Latitude (°)')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'gps_latitude_vs_interpolated.png'))
    plt.show()

    # Plot Longitude
    plt.figure(figsize=(12, 6))
    plt.scatter(original_times / 1e9, original_lons, label='Measured', alpha=0.5)
    plt.plot(interpolated_times / 1e9, interpolated_lons, 'r-', label='Interpolated', alpha=0.7)
    plt.title('Longitude - Measured vs Interpolated')
    plt.xlabel('Time (s)')
    plt.ylabel('Longitude (°)')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'gps_longitude_vs_interpolated.png'))
    plt.show()

    # Plot GPS trajectory
    plt.figure(figsize=(12, 6))
    plt.scatter(original_lons, original_lats, label='Measured', alpha=0.5)
    plt.plot(interpolated_lons, interpolated_lats, 'r-', label='Interpolated', alpha=0.7)
    
    plt.title('GPS Trajectory - Original vs Interpolated')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'gps_trajectory_vs_interpolated.png'))
    plt.show()

def plot_xy_interpolation(original_gps, interpolated_gps, ref_lat, ref_lon, file_path = None):
    # Convert original GPS to XY
    original_times = np.array([t for t, _, _, _ in original_gps]) - original_gps[0][0]
    original_xy = np.array([latlon_to_xy(lat, lon, ref_lat, ref_lon) for _, lat, lon, _ in original_gps])

    # Convert interpolated GPS to XY
    interpolated_times = np.array([t for t, _, _, _ in interpolated_gps]) - interpolated_gps[0][0]
    interpolated_xy = np.array([latlon_to_xy(lat, lon, ref_lat, ref_lon) for _, lat, lon, _ in interpolated_gps])

    # Plot X coordinate
    plt.figure(figsize=(12, 6))
    plt.scatter(original_times / 1e9, original_xy[:, 0], label='Measured', alpha=0.5)
    plt.plot(interpolated_times / 1e9, interpolated_xy[:, 0], 'r-', label='Interpolated', alpha=0.7)
    plt.title('X Coordinate - Measured vs Interpolated')
    plt.xlabel('Time (s)')
    plt.ylabel('X (m)')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'xy_x_coordinate_vs_interpolated.png'))
    plt.show()

    # Plot Y coordinate
    plt.figure(figsize=(12, 6))
    plt.scatter(original_times / 1e9, original_xy[:, 1], label='Measured', alpha=0.5)
    plt.plot(interpolated_times / 1e9, interpolated_xy[:, 1], 'r-', label='Interpolated', alpha=0.7)
    plt.title('Y Coordinate - Measured vs Interpolated')
    plt.xlabel('Time (s)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'xy_y_coordinate_vs_interpolated.png'))
    plt.show()

    # Plot XY trajectory
    plt.figure(figsize=(12, 6))
    plt.scatter(original_xy[:, 0], original_xy[:, 1], label='Measured', alpha=0.5)
    plt.plot(interpolated_xy[:, 0], interpolated_xy[:, 1], 'r-', label='Interpolated', alpha=0.7)
    plt.title('X/Y Trajectory - Measured vs Interpolated')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')
    plt.minorticks_on()
    if file_path:
        save_path = os.path.join(base_path,file_path)
        # Create the folder, including any necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'xy_trajectory_vs_interpolated.png'))
    plt.show()

