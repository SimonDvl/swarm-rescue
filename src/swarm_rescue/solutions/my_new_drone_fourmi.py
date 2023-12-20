import math
from copy import deepcopy
from typing import Optional

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign


class MyNewDroneFourmi(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data

    def control(self):
        """
        In this example, we only use the sensors sensor and the communication to move the drone
        The idea is to make the drones move like a school of fish.
        The sensors will help avoid running into walls.
        The communication will allow to know the position of the drones in the vicinity, to then correct its own
        position to stay at a certain distance and have the same orientation.
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0}

        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())

        return command_lidar

    def process_lidar_sensor(self, the_lidar_sensor):
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller = 0.5

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

        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        far_angle = normalize_angle(far_angle)

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        collision = False
        if size != 0 and min_dist < 50:
            collision = True
            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller

        return command, collision