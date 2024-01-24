from enum import Enum
import math
from copy import deepcopy
from typing import Optional

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor


class MyNewDroneFourmi(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4
        INITIALIZING = 5


    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.state = self.Activity.INITIALIZING
        self.initialized = False

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data

    def control(self):
        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor(self.semantic())
        print(self.state)
        # TRANSITIONS OF THE STATE MACHINE

        if self.state is self.Activity.INITIALIZING and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        if self.state is self.Activity.INITIALIZING and self.initialized:
            self.state = self.Activity.SEARCHING_WOUNDED

        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.state.SEARCHING_RESCUE_CENTER

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.state.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        '''
        print("state: {}, can_grasp: {}, grasped entities: {}, found wounded: {}".format(self.state.name,
                                                                                         self.base.grasper.can_grasp,
                                                                                         self.base.grasper.grasped_entities,
                                                                                         found_wounded))
        print("state: {}, can_grasp: {}, grasped entities: {}, found wounded: {}".format(self.state.name,
                                                                                        self.base.grasper.can_grasp,
                                                                                        self.base.grasper.grasped_entities,
                                                                                        found_wounded))
        '''
        ##########
        # COMMANDS FOR EACH STATE
        # Searching by following wall, but when a rescue center or wounded person is detected, we use a special command
        ##########

        if self.state is self.Activity.INITIALIZING:
            command = self.control_wall()
            command = {"forward": 1.0,
                       "lateral": 0.0,
                       "rotation": 0.0,
                       "grasper": 0}

        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_wall()
            print(command)
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_wall()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1

        return command

    # def update_command(self):

    #     the_lidar_sensor = self.lidar()
    #     values = the_lidar_sensor.get_sensor_values()
    #     distance_to_wall = 100  # Distance cible par rapport au mur
    #     gain = 0.1  # Facteur de correction proportionnel
    #     command = {"forward": 1.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

    #     # Trouver l'angle où le mur est le plus proche
    #     min_index = np.argmin(values)
    #     closest_distance = values[min_index]

    #     # Calculer l'erreur par rapport à la distance cible
    #     error = closest_distance - distance_to_wall

    #     # Correction proportionnelle pour ajuster la distance au mur
    #     lateral_correction = gain * error

    #     # Mettre à jour la commande
    #     command["lateral"] = lateral_correction

    #     # Rotation pour maintenir le mur à droite
    #     command["rotation"] = -0.1  # Ajustez cette valeur selon les besoins

    #     return command
    
    def control_wall(self):
        # command_straight = {"forward": 0.3,
        #                     "lateral": 0.0,
        #                     "rotation": 0.0,
        #                     "grasper": 0}

        # command_right = {"forward": 0.5,  # freiner l'inertie
        #                  "lateral": -0.9,
        #                  "rotation": -0.4,
        #                  "grasper": 0}

        # command_turn = {"forward": 0.0,
        #                 "lateral": 0.0,
        #                 "rotation": 0.5,  # increase if too slow but it should ensure that we don't miss the moment when the wall is on the right
        #                 "grasper": 0}

        # command_left = {"forward": 0.0,
        #                 "lateral": 0.0,
        #                 "rotation": 1,
        #                 "grasper": 0}

        command = {"forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0.0}
        
        touch_array = self.lidar_wall_touch()
        touch_counter = len(touch_array)
        
        # Initialization -> Go straight to wall when the drone doesn't touch any wall i.e. case when he is lost
        if touch_counter == 0:
            command["forward"] = 1.0
        
        # When the drone touches a wall, first the drone must put the wall on his right (rotation if necessary) and then go straight forward
        elif touch_counter == 1.0:
            self.initialized = True
            # Case when the drone is parallel to the wall 
            if touch_array[0]<-0.6*np.pi/2 and touch_array[0]>-1.1*np.pi/2 : 
                command["forward"] = 0.7
                # Get further to the wall if too close
                if min(self.lidar().get_sensor_values())<30:
                    command['lateral'] = 0.3

            elif touch_array[0]>-0.6*np.pi/2 and touch_array[0]<= 0:
                command['forward'] = 0.0
                command["rotation"] = 1.0
                command["lateral"] = 0.0
            elif touch_array[0]>-np.pi and touch_array[0]<= -1.1*np.pi/2:
                command['forward'] = 0.0
                command["rotation"] = -1.0
                command['lateral'] = -0.2
            elif touch_array[0]>0 and touch_array[0]<=np.pi/2:
                command["rotation"] = 1.0
            else:
                command["rotation"] = -1.0
                
        # When the drone is in a corner        
        else: 
            self.initialized = True
            command["forward"] = 0.0
            command["rotation"] = 1.0
            
        # Avoid collision anyway
        
        collision_dist = min(self.lidar().get_sensor_values())<50
        
        # collision_angle = self.lidar().ray_angles[np.argmin(collision_dist)]
        # if collision_dist<50 and collision_angle<0:
        #     command["rotation"] = 1.0
        #     command["forward"] = 1.0
        # # else:
        # #     command["rotation"] = 1.0
        # #     command["forward"] = 1.0
            
        return command
            

    def process_semantic_sensor(self, the_semantic_sensor):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 0.8,
                   "lateral": 0.0,
                   "rotation": 0.0}

        angular_vel_controller_max = 1.0

        detection_semantic = the_semantic_sensor.get_sensor_values()
        best_angle = 1000

        found_wounded = False
        is_near_wounded = False
        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    self.initialized = True
                    best_angle = data.angle
                    is_near_wounded = (data.distance < 50)

        is_near_rescue = False
        found_rescue_center = False
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    best_angle = data.angle
                    is_near_rescue = (data.distance < 70)

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

        if found_rescue_center and is_near_rescue:
            command["forward"] = 0
            command["rotation"] = -1.0

        if found_wounded and is_near_wounded:
            command["forward"] = 0
            command["rotation"] = -1.0

        return found_wounded, found_rescue_center, command
    
    # def process_lidar_sensor(self, the_lidar_sensor):
    #     command = {"forward": 1.0,
    #                "lateral": 0.0,
    #                "rotation": 0.0}
    #     angular_vel_controller = 0.5

    #     values = the_lidar_sensor.get_sensor_values()

    #     if values is None:
    #         return command, False

    #     ray_angles = the_lidar_sensor.ray_angles
    #     size = the_lidar_sensor.resolution

    #     far_angle_raw = 0
    #     near_angle_raw = 0
    #     min_dist = 1000
    #     if size != 0:
    #         # far_angle_raw : angle with the longer distance
    #         far_angle_raw = ray_angles[np.argmax(values)]
    #         min_dist = min(values)
    #         # near_angle_raw : angle with the nearest distance
    #         near_angle_raw = ray_angles[np.argmin(values)]

    #     far_angle = far_angle_raw
    #     # If far_angle_raw is small then far_angle = 0
    #     if abs(far_angle) < 1 / 180 * np.pi:
    #         far_angle = 0.0

    #     near_angle = near_angle_raw
    #     far_angle = normalize_angle(far_angle)

    #     # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
    #     collision = False
    #     if size != 0 and min_dist < 50:
    #         collision = True
    #         if near_angle > 0:
    #             command["rotation"] = -angular_vel_controller
    #         else:
    #             command["rotation"] = angular_vel_controller

    #     return command, collision
    
    def lidar_wall_touch(self):
        """
        Returns an array with the angle(s) for which a wall is the closest (can be 0,1,2)
        """

        the_lidar_sensor = self.lidar()
        values = the_lidar_sensor.get_sensor_values()
        values_copy = values.copy()
        
        angles = np.zeros(2)
        if values is None:
            return angles


        ray_angles = the_lidar_sensor.ray_angles
        ray_angles_copy = ray_angles.copy()

        # Get the two lower values and get the corresponding angles
        min_dist_1 = min(values_copy)
        near_angle_1 = ray_angles_copy[np.argmin(values_copy)]
        
        # Remove the minimum value and the corresponding angle
        values_copy = np.delete(values_copy,np.argmin(values_copy))
        ray_angles_copy = np.delete(values_copy,np.argmin(values_copy))
        
        min_dist_2 = min(values_copy)
        near_angle_2 = ray_angles[np.argmin(values_copy)]

        # Store values in arrays
        near_angle = np.array([near_angle_1,near_angle_2])
        min_dist = np.array([min_dist_1, min_dist_2])
        
        # Check if we are close to a wall and keep the corresponding values
        threshold = 100
        for i in range(1):
            if min_dist[i] > threshold:
                near_angle = np.delete(near_angle, i)
        
        # Check if the two angles are close to each other (less than 20°)   
        if len(near_angle)==2:
            near_angle_normalized = normalize_angle(near_angle)
            angle_diff = abs(near_angle_normalized[0]-near_angle_normalized[1])
            if angle_diff <20 * np.pi/180:
                near_angle = np.delete(near_angle,0)
        print(near_angle)
        return(near_angle)

    # def lidar_wall_touch(self):
    #     """
    #     Returns an array with the angle(s) for which a wall is the closest and the number of touch (can be 0, 1, or 2)
    #     """

    #     the_lidar_sensor = self.lidar()
    #     values = the_lidar_sensor.get_sensor_values()

    #     touch_counter = 0
    #     angles = np.zeros(3)
    #     if values is None:
    #         return np.zeros(3)

    #     ray_angles = the_lidar_sensor.ray_angles

    #     # Get the angles from the two lower values
    #     sorted_indices = np.argpartition(values, 2)[:2]
    #     print("sorted indices", sorted_indices)
    #     print("angles values", values[sorted_indices])
    #     # Check if a wall is close according to a threshold
    #     threshold = 100
    #     for j, i in enumerate(sorted_indices[:2]):
    #         if values[i] < threshold:
    #             touch_counter += 1
    #             angles[j] = ray_angles[i]

    #     if touch_counter == 2 and abs(angles[0] - angles[1]) < (20*2*np.pi/180):
    #         # Consecutive angles are assimilated with only one touch
    #         touch_counter = 1
    #         angles = angles[1:]

    #     return np.concatenate([angles, [touch_counter]])
