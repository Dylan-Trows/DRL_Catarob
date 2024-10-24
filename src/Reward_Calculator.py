import numpy as np
import math
from numba import jit


class RewardCalculator:
    def __init__(self, alpha=1, beta=1, max_speed=2):
        self.alpha = alpha
        self.beta = beta
        self.max_speed = max_speed
        self.previous_distance = None
        self.maximum_distance = None            # max straight line distance from start to target 
        self.total_reward = 0
        self.total_distance = 0
        self.prev_x = None
        self.prev_y = None
        
    jit(nopython=True)
    def calculate_reward(self, current_state, action, waypoint):
        if waypoint is None or current_state is None or action is None:
            return 0.0

        current_x, current_y = current_state[:2]
        
        if self.prev_x is None:
            self.prev_x = current_x
            self.prev_y = current_y
        
        step_distance = round(math.sqrt((current_x - self.prev_x)**2 + (current_y - self.prev_y)**2), 2)
        #print("Step distance :",step_distance)
        self.total_distance += step_distance

        heading_error = current_state[3]

        """# Calculate current distance to waypoint
        current_distance = math.sqrt((current_x - waypoint[0])**2 + (current_y - waypoint[1])**2)
        """ 
        current_distance = current_state[4]

        # Initialize maximum_distance and previous_distance if not set
        if self.maximum_distance is None:
            self.maximum_distance = current_distance
        if self.previous_distance is None:
            self.previous_distance = current_distance

        # Distance Reward
        """l_arc = self.previous_distance * math.radians(abs(heading_error))"""
        l_arc = current_state[5]
        normalized_d_target = current_distance / self.maximum_distance
        normalized_l_arc = l_arc / (math.pi * current_distance)
        w_arc, w_dist = 0.5, 0.5
        distance = w_arc * normalized_l_arc + w_dist * normalized_d_target
        #distance_reward = -self.alpha * math.tanh(distance - 1)
        # simpler reward using function f(x) = 1 - x 
        distance_reward = 1 - distance

        # Velocity reward
        """forward_speed = action[0]"""
        speed = current_state[6]
        speed_heading_reward = 0.5*(speed * math.cos(math.radians(heading_error))) / self.max_speed
        movement_reward = self.beta * speed_heading_reward

        # Calculate total reward
        reward = distance_reward + movement_reward 
        
        # Check if task is finished (reached the waypoint)
        if math.sqrt((current_x - waypoint[0])**2 + (current_y - waypoint[1])**2) < 1.0:
            
            efficiency = ( (self.maximum_distance - current_distance) / (self.total_distance) )
            print("Max distance: ", self.maximum_distance,", Total Distance: ",self.total_distance,", Efficiency: ",efficiency)
            reward += 200*efficiency

        # Update previous_distance for next iteration
        self.previous_distance = current_distance
        self.total_reward += reward
        self.prev_x = current_x
        self.prev_y = current_y

        return reward