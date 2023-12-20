"""
First iteration of Drone_Wall
The Drone will follow the walls to find injured
"""

from typing import Optional, List, Type
import numpy as np
from enum import Enum

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle


class MyAntDrone(DroneAbstract):
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
                         should_display_lidar=False,
                         **kwargs)
        self.state = self.Activity.INITIALIZING
        self.initialized = False

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def touch_acquisition(self):
        """"
        Returns nb of touches (0|1|2) and Vector indicating triggered captors
        """
        zeros = np.zeros(13)

        if self.touch().get_sensor_values() is None:
            return zeros

        nb_touches = 0
        detection = self.touch().get_sensor_values()

        # Getting the two highest values from detection
        max = np.maximum(detection[0], detection[1])
        second_max = np.minimum(detection[0], detection[1])
        n = len(detection)
        for i in range(2, n):
            if detection[i] > max:
                second_max = max
                max = detection[i]
            elif detection[i] > second_max and max != detection[i]:
                second_max = detection[i]
            elif max == second_max and second_max != detection[i]:
                second_max = detection[i]

        # The two highest values from the list are changed to 1 if they are higher than threshold
        # Other values are changed to 0
        for i in range(n):
            threshold = 0.95
            if detection[i] > threshold and detection[i] >= second_max:
                detection[i] = 1
                nb_touches += 1

            else:
                detection[i] = 0

        # Consecutive 1s are counted as only one touch
        for i in range(n-1):
            if detection[i] == 1 and detection[i+1] == 1:
                detection[i] = 0
                nb_touches -= 1

        if nb_touches > 2:
            return zeros

        # Number of touches is set as last value of the list (last value not useful)
        detection[-1] = nb_touches
        return detection

    def control(self):

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor(
            self.semantic())

        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

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

    def control_wall(self):

        command_straight = {"forward": 1.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        command_right = {"forward": 0.5,  # freiner l'inertie
                         "lateral": -0.9,
                         "rotation": -0.4,
                         "grasper": 0}

        command_turn = {"forward": 1.0,
                        "lateral": 0.0,
                        "rotation": 1.0,  # increase if too slow but it should ensure that we don't miss the moment when the wall is on the right
                        "grasper": 0}

        command_left = {"forward": 0.2,
                        "lateral": 0.0,
                        "rotation": 1.0,
                        "grasper": 0}

        touch_array = self.touch_acquisition()

        # Initialization -> Go straight to wall

        # when the drone doesn't touch any wall i.e. case when he is lost
        if touch_array[-1] == 0.0:
            return command_right

        # when the drone touches a wall, first the drone must put the wall on his right (rotation if necessary) and then go straight forward
        elif touch_array[-1] == 1.0:
            # which indices correspond to the ray at 90 degrees on the right ???
            self.initialized = True
            if touch_array[1] + touch_array[2] + touch_array[3] >= 1:
                return command_straight

            else:
                return command_turn
        elif touch_array[-1] == 2.0:  # when the drone is in a corner
            self.initialized = True
            return command_left

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
