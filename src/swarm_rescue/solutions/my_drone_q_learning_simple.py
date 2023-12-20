import numpy as np
import math
from copy import deepcopy
from typing import Optional

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign


class MyDroneQLearningSimple(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.num_actions = 4 #TODO : update depending on the action number
        self.num_states = 300

        # Initialize Q-table
        self.q_table = np.zeros((self.num_states, self.num_actions))

        # Reward
        self.total_reward = 0

        # Memory for the past visited positions
        self.visited_positions = set()

        
    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data
    
    def control(self):
        current_state = self.get_current_state()
        chosen_action = self.epsilon_greedy_policy(current_state)
        command = self.execute_action(chosen_action)

        reward = self.get_reward()
        self.total_reward += reward
        next_state = self.get_next_state()

        self.update_q_table(current_state, chosen_action, reward, next_state)

        return command

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.choice(self.num_actions)
        else:
            # Choose the action with the largest Q
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q_value = self.q_table[state, action]
        best_next_q_value = np.max(self.q_table[next_state])
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * best_next_q_value - current_q_value)
        self.q_table[state, action] = new_q_value

    def execute_action(self, action):
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0}
        #TODO : find the optimal number of action/ command
        if action == 0:
            command["forward"] = 0.4
        elif action == 1:
            command["rotation"] = 1
        elif action == 2:
            command["forward"] = -0.1
        elif action == 3:
            command["rotation"] = -1
        # elif action == 4:
        #     command["forward"] = 0.8
        # elif action == 5:
        #     command["rotation"] = 0.8
        # elif action == 6:
        #     command["forward"] = -0.8
        # elif action == 7:
        #     command["rotation"] = -0.8
        # elif action == 8:
        #     command["forward"] = -0.6
        # elif action == 9:
        #     command["rotation"] = -0.6
        # elif action == 10:
        #     command["forward"] = -0.4
        # elif action == 11:
        #     command["rotation"] = -0.4

        return command

    def update_visited_positions(self):
        current_position = self.gps().get_sensor_values()
        self.visited_positions.add(current_position)

    def is_near_visited_position(self, radius):
        current_position = self.gps().get_sensor_values()[1]
        for visited_position in self.visited_positions:
            distance = self.calculate_distance(current_position, visited_position)
            if distance < radius:
                return True
        return False

    def calculate_distance(self, position1, position2):
        x1, y1 = position1
        x2, y2 = position2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    
    def get_current_state(self):
        # Current state is defined as the minimum value of the lidar sensor
        #TODO : try with another kind of state, for instance angle of the closest wall ?
        lidar_values = self.lidar().get_sensor_values()
        return np.argmin(lidar_values)


    def get_reward(self):
        # Get a reward if the drone explores a new position and get a penalty when it gets too close to a wall
        #TODO : find the ideal radius
        reward = 0
        current_state = self.get_current_state()
        
        if current_state>1 : 
            reward = 1

        if not self.is_near_visited_position(radius=150):
            reward = 0.2

        return reward

    def get_next_state(self):
        current_state = self.get_current_state()
        chosen_action = self.epsilon_greedy_policy(current_state)
        self.execute_action(chosen_action)
        next_state = self.get_current_state()
        return next_state