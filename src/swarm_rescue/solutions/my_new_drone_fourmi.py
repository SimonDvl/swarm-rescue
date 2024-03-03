from enum import Enum
import math
from copy import deepcopy
from typing import Optional

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.mapping import OccupancyGrid
from spg_overlay.utils.pose import Pose
from solutions.astar import astar



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
        COMING_BACK = 6


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
        
        self.iteration: int = 0
        self.estimated_pose = Pose()

        resolution = 30 # Resolution de la grille 
        #TODO: essayer de faire varier la résolution de la grille pour accélérer l'algo A* 
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())
        
        
        self.rescue_center = (0,0)
        self.target_distance = 10
        self.obstacle_distance = 5 
        self.integral_error = 0.0
        self.last_touch_array = np.zeros(2)
        self.victim_position = 0
        self.trajectory = []
        self.best_path = []
        self.best_path_back =[]
        self.best_path_copy = []
        
        
        
    def define_message_for_all(self):
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data
    
    def control(self):
        print(self.state)
        """
        Control the drone, update the state machine and compute the command depending on the state
        """

        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor(self.semantic())
        reached_location = self.process_reached_position()
        # TRANSITIONS OF THE STATE MACHINE
        if self.state is self.Activity.INITIALIZING and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.INITIALIZING and self.initialized:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.Activity.SEARCHING_WOUNDED

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
        
        #TODO : Quand le drone a deposé la victime au rescue center, faire en sorte que le drone reparte chercher des victimes et repasser le grasper à 0
        #TODO : Idée : reparcourir le best path en sens inverse pour le faire retourner là où il a recup la victime et éviter de suivre à nouveau le mur, puis reprendre l'explo fourmi
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.COMING_BACK

        elif self.state is self.Activity.COMING_BACK and reached_location:
            print(f"Trying to transition from COMING_BACK to SEARCHING_WOUNDED. State: {self.state}")

            self.state = self.Activity.SEARCHING_WOUNDED
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
            self.best_path=[]
            command = self.control_wall()
            command["grasper"] = 0

        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1
            self.victim_position = self.gps().get_sensor_values()
            

        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            # Compute best path to come back to rescue center
            if len(self.best_path) == 0 : 
                #TODO : ajouter le cas ou il ne trouve pas de meilleur chemin => control wall
                self.best_path = self.compute_best_path()
                self.best_path_copy = self.best_path.copy()
                self.best_path_back = self.best_path_copy.copy()[::-1]
            command = self.follow_trajectory(self.best_path)
            # command = self.control_wall()
            command["grasper"] = 1

        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1
            
            
        elif self.state is self.Activity.COMING_BACK:
            command = self.follow_trajectory(self.best_path_back)
            command["grasper"] = 0
        ###### Update the map ######
        #TODO : 1) faire en sorte qu'un seul drone actualise la map quand d'autres drones sont à portée puis la partage sinon c'est lent
        #TODO : 2) problème de detection du lidar: considère les autres drones/ victimes comme des murs (pb pour l'algo a*)
        # increment the iteration counter
        self.iteration += 1

        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()), self.measured_compass_angle())

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid, self.estimated_pose, title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid, self.estimated_pose, title="zoomed occupancy grid")

        
        ###### Keep de location of the rescue center in memory ######
        
        the_semantic_sensor = self.semantic()
        detection_semantic = the_semantic_sensor.get_sensor_values()
        if self.rescue_center == (0,0) : 
            #TODO : modifier pour que ça soit vraiment la position du rescue center et pas sa position de départ
            initial_position = self.gps().get_sensor_values()
            self.rescue_center = (initial_position[0], initial_position[1])
            # for data in detection_semantic : 
            #     if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
            #         self.rescue_center = self.compute_position_from_distance_angle(data.distance, data.angle)
            #         print("RESCUE CENTER POSITION", self.rescue_center, 'DRONE POSITION', self.gps().get_sensor_values(),"DATA", DeprecationWarning)
                
        
    
        return command
    
    def process_reached_position(self):
        reached_location = False
        current_position = self.gps().get_sensor_values()
        ##TODO : do the case when current_position not available
        vector_to_target = np.array(self.victim_position) - np.array(current_position)
        distance_to_target = np.linalg.norm(vector_to_target)
        if distance_to_target<100:
                reached_location = True
        return reached_location
    
    def compute_best_path(self):
        """
        Compute the best path to come back to rescue center using the A* algorithm 
        
        Returns a list of position tuple
        """
        
        drone_position = self.gps().get_sensor_values()
        drone_position_on_grid_x, drone_position_on_grid_y  = self.grid._conv_world_to_grid(drone_position[0],drone_position[1]) 
        start_point = (drone_position_on_grid_x, drone_position_on_grid_y)
        
        rescue_center_on_grid_x, rescue_center_on_grid_y= self.grid._conv_world_to_grid(self.rescue_center[0], self.rescue_center[1])
        end_point = (rescue_center_on_grid_x,rescue_center_on_grid_y)
    
        best_path_grid = astar(start_point, end_point, self.grid.grid)
        best_path_world=[]
        for element in best_path_grid:
            x_real_world,y_real_world = self.grid._conv_grid_to_world(element[0],element[1])
            best_path_world.append((x_real_world,y_real_world))
                    
        return best_path_world
    
    def compute_position_from_distance_angle(self, distance, angle):
        """
        Compute the position from a distance, an angle and from the position of the drone
        """
        # Get drone position
        drone_position = self.gps().get_sensor_values()
        x = drone_position[0]
        y = drone_position[1]
        
        # Compute the position of rescue center 
        x_rescue_center = distance*np.cos(angle) + x
        y_rescue_center = distance*np.sin(angle) + y
        
        return(x_rescue_center,y_rescue_center)
    
    def control_wall(self):
        """
        Compute the command to make the drone follow the right wall while avoiding collision with it
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0.0}

        touch_array = self.lidar_wall_touch()
        touch_counter = len(touch_array)

        # Initialization -> Go straight to the wall when the drone doesn't touch any wall (case when it is lost)
        if touch_counter == 0:
            command["forward"] = 1.0

        # When the drone touches a wall, apply PI controller
        elif touch_counter == 1.0:
            self.initialized = True

            # Calculate the error term (difference between the desired angle and the current touch angle)
            error = touch_array[0] + np.pi / 2  # Adjusting to have 0 at the right side
            rotation_gain = 1.0
            integral_gain = 0.01
            

            # Proportional term
            proportional = rotation_gain * error

            # Integral term (cumulative error over time)
            self.integral_error += error  # Update the integral error

            # Combine proportional and integral terms for rotation adjustment
            rot = min(1.0, max(-1.0, proportional + integral_gain* self.integral_error))
            command["rotation"] = rot
            
            # if rot < 0:
            #     command["lateral"] = 1.0*rot
            
            # Adjust lateral command for maintaining distance from the wall
            lateral_gain = 0.05 # Adjust 
            lateral_distance = 20 # Desired lateral distance from the wall

            lateral_error = lateral_distance - min(self.lidar().get_sensor_values())
            command["lateral"] = max(-1.0, min(1.0,lateral_gain * lateral_error))

            # Adjust forward command for smoother movement
            if abs(command["rotation"])==1.0:
                command["forward"] = 0.2 # if we need to turn a lot
                command["lateral"] = 1.0 * rot
            else:
                command["forward"] = 1.0

        # When the drone is in a corner
        else:
            self.initialized = True
            command["forward"] = 0.0
            command["rotation"] = 1.0

        # Store the current touch_array for future comparison
        self.last_touch_array = touch_array

        return command
   
    def process_semantic_sensor(self, the_semantic_sensor):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        command = {"forward": 1.0,
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
        ##TODO add the case when the drone has a wall on the right and on the left to tell him to continue ot follow the wall on the right
        
        return(near_angle)
    
    
    def follow_trajectory(self, trajectory):
        """
        Follow a predefined trajectory using proportional control
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0.0}

        if trajectory is None:
            return command  # No trajectory to follow i.e. when the best path algorithm did not work

        # Get current drone position and orientation
        current_position = self.gps().get_sensor_values()
        current_orientation = self.compass().get_sensor_values()

        # Get the first point in the trajectory
        target_position = trajectory[0]

        # Compute the vector from the drone to the target
        vector_to_target = np.array(target_position) - np.array(current_position)

        # Calculate the angle between the drone's orientation and the vector to the target
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])

        # Proportional control for rotation
        angle_difference = normalize_angle(angle_to_target - current_orientation)
        rotation_gain = 0.6
        command["rotation"] = max(-1.0,min(1.0,rotation_gain * angle_difference))

        # Proportional control for forward movement
        distance_to_target = np.linalg.norm(vector_to_target)
        forward_gain = 2.0
        command["forward"] = max(-1.0,min(1.0, forward_gain * distance_to_target))
        # Remove the point from the trajectory list when close to it
        if distance_to_target < 70:
            trajectory.pop(0) 
        
        # Repulsive force to avoid the walls
        if min(self.lidar().get_sensor_values())<20:
            touch_array = self.lidar_wall_touch()
            if touch_array[0]>=np.pi/4 and touch_array[0]<=np.pi:
                command["lateral"] = -0.5
            elif touch_array[0]<=-np.pi/4 and touch_array[0]>=-np.pi:
                command["lateral"] = +0.5
            elif touch_array[0]<=0 and touch_array[0]>=-np.pi/4:
                command['forward'] = -1.0
                command['rotation'] = 1.0
            else:
                command['forward'] = -1.0
                command['rotation'] = -1.0
                
        return command