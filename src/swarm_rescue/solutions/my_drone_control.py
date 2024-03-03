import math
from copy import deepcopy
from typing import Optional
import pandas as pd
import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign


class MyDroneControl(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.counter = 0
        self.df = pd.DataFrame()
    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data

    def control(self):
        if self.counter%10 == 0 :
            command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.2}
        else : 
            command = {"forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0}
        # print(self.compass().get_sensor_values())
        
        command =  {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}
        if self.counter<=10:
            command = {"forward": 1.0,
                        "lateral": 0.0,
                        "rotation": 0.0}
        self.counter+=1
        values = self.lidar().get_sensor_values()
        ray_angles = self.lidar().ray_angles
        near_angle = ray_angles[np.argmin(values)]
        far_angle = ray_angles[np.argmax(values)]


        compass_values = self.compass().get_sensor_values()

        error = far_angle-compass_values

        if  abs(error)> 0.2:
            normalize_near_angle = normalize_angle(near_angle)
            print(near_angle, normalize_near_angle)
            
        return {"forward": 0.0,
                        "lateral": 0.5,
                        "rotation": 0.0}
    
"""
Rotation : 
Angle = f(rotation_command) ~ 10 * rotation_command
"""
