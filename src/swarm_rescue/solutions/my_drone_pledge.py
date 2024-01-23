from enum import Enum
import math
from copy import deepcopy
from typing import Optional

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor


class MyDronePledge(DroneAbstract):
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
            command_lidar= {"forward": 1.0,
                       "lateral": 0.0,
                       "rotation": 0.0,
                       "grasper": 0}

        elif self.state is self.Activity.SEARCHING_WOUNDED:
            command_lidar = self.control_wall()
            command_lidar["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command_lidar = command_semantic
            command_lidar["grasper"] = 1

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command_lidar = self.control_wall()
            command_lidar["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command_lidar = command_semantic
            command_lidar["grasper"] = 1

        collision_lidar = self.process_lidar_sensor(self.lidar())[1]
        found, command_comm = self.process_communication_sensor()

        alpha = 0.4
        alpha_rot = 0.75

        if collision_lidar:
            alpha_rot = 0.1

        command= {"forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0}
        # The final command  is a combination of 2 commands
        command["forward"] = \
            alpha * command_comm["forward"] \
            + (1 - alpha) * command_lidar["forward"]
        command["lateral"] = \
            alpha * command_comm["lateral"] \
            + (1 - alpha) * command_lidar["lateral"]
        command["rotation"] = \
            alpha_rot * command_comm["rotation"] \
            + (1 - alpha_rot) * command_lidar["rotation"]
        command["grasper"] = command_lidar["grasper"]
        print(command_lidar)
        return command_lidar
        
    def process_lidar_sensor(self, the_lidar_sensor):
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                    "grasper" : 0.0}
        angular_vel_controller = 1

        values = the_lidar_sensor.get_sensor_values()
        
        if values is None:
            return command, False

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution
        far_angle_raw = 0
        near_angle_raw = 0
        min_dist = 1000
        if size != 0:
            # far_angle_raw : angle with the longer distance
            far_angle_raw = ray_angles[np.argmax(values)]
            min_dist = min(values)
            # near_angle_raw : angle with the nearest distance
            near_angle_raw = ray_angles[np.argmin(values)]
            if min_dist <150:
                self.initialized = True
        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        far_angle = normalize_angle(far_angle)
        
        normalized_near_angle = normalize_angle(near_angle)
        print(normalized_near_angle, near_angle)

        desired_angle = np.pi/2
        error = normalized_near_angle - desired_angle
        #TODO : faire une fonction qui permet d'ajuster la rotation en fonction de l'angle duquel le drone doit tourner pour mettre le mur sur sa droite pour que ça soit plus rapide
        if near_angle>=0 and near_angle<=np.pi/2:
            # Pure rotation on the left to put the wall on its right
            command = {"forward": 0.1,
                       "lateral": 0.0,
                       "rotation": 1.0,
                       "grasper": 0}
        elif near_angle>np.pi/2:
            # Pure rotation on the right to put the wall on its right
            command = {"forward": 0.1,
                       "lateral": 0.0,
                       "rotation": -1.0,
                       "grasper": 0}
        elif near_angle<0 and near_angle>=-np.pi/2:
            # Slight left turn to keep wall on the right
            command = {"forward": 1.0,
                       "lateral": 0.0,
                       "rotation": 1.0,
                       "grasper": 0}
        else : 
            command = {"forward": 0.1,
                       "lateral": 0.0,
                       "rotation": -1.0,
                       "grasper": 0} 

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        collision = False
        if size != 0 and min_dist < 30:
            collision = True
            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller
        
        return command, collision
    
     
    def control_wall(self):
        """
        Follow the right wall with a certain distance, based on the Pledge algorithm
        """
        the_lidar_sensor = self.lidar()
        command, collision = self.process_lidar_sensor(the_lidar_sensor)
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
    
    def process_communication_sensor(self):
        found_drone = False
        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}

        if self.communicator:
            received_messages = self.communicator.received_messages
            nearest_drone_coordinate1 = (
                self.measured_gps_position(), self.measured_compass_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_gps_position()[0]
                dy = other_position[1] - self.measured_gps_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone, command_comm

            # If we found at least 2 drones
            if found_drone and len(received_messages) >= 2:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                (nearest_position2, nearest_angle2) = nearest_drone_coordinate2
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle2 = normalize_angle(
                    nearest_angle2 - self.measured_compass_angle())
                # The mean of 2 angles can be seen as the angle of a vector, which
                # is the sum of the two unit vectors formed by the 2 angles.
                diff_angle = math.atan2(0.5 * math.sin(diff_angle1) + 0.5 * math.sin(diff_angle2),
                                        0.5 * math.cos(diff_angle1) + 0.5 * math.cos(diff_angle2))

            # If we found only 1 drone
            elif found_drone and len(received_messages) == 1:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle = diff_angle1

            # if you are far away, you get closer
            # heading < 0: at left
            # heading > 0: at right
            # base.angular_vel_controller : -1:left, 1:right
            # we are trying to align : diff_angle -> 0
            command_comm["rotation"] = sign(diff_angle)

            # Desired distance between drones
            desired_dist = 60

            d1x = nearest_position1[0] - self.measured_gps_position()[0]
            d1y = nearest_position1[1] - self.measured_gps_position()[1]
            distance1 = math.sqrt(d1x ** 2 + d1y ** 2)

            d1 = distance1 - desired_dist
            # We use a logistic function. -1 < intensity1(d1) < 1 and  intensity1(0) = 0
            # intensity1(d1) approaches 1 (resp. -1) as d1 approaches +inf (resp. -inf)
            intensity1 = 2 / (1 + math.exp(-d1 / (desired_dist * 0.5))) - 1

            direction1 = math.atan2(d1y, d1x)
            heading1 = normalize_angle(direction1 - self.measured_compass_angle())

            # The drone will slide in the direction of heading
            longi1 = intensity1 * math.cos(heading1)
            lat1 = intensity1 * math.sin(heading1)

            # If we found only 1 drone
            if found_drone and len(received_messages) == 1:
                command_comm["forward"] = longi1
                command_comm["lateral"] = lat1

            # If we found at least 2 drones
            elif found_drone and len(received_messages) >= 2:
                d2x = nearest_position2[0] - self.measured_gps_position()[0]
                d2y = nearest_position2[1] - self.measured_gps_position()[1]
                distance2 = math.sqrt(d2x ** 2 + d2y ** 2)

                d2 = distance2 - desired_dist
                intensity2 = 2 / (1 + math.exp(-d2 / (desired_dist * 0.5))) - 1

                direction2 = math.atan2(d2y, d2x)
                heading2 = normalize_angle(direction2 - self.measured_compass_angle())

                longi2 = intensity2 * math.cos(heading2)
                lat2 = intensity2 * math.sin(heading2)

                command_comm["forward"] = 0.5 * (longi1 + longi2)
                command_comm["lateral"] = 0.5 * (lat1 + lat2)

        return found_drone, command_comm
