import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BirdsEyeViewStaticObstacleLocationsEnvironment(gym.Env):
    def __init__(self, TIME_STEPS=200, 
                 initialize_state_randomly=True, 
                 SAMPLING_TIME_SECONDS=0.05, 
                 INITIAL_STATE=np.array([0,0,0,1], dtype=np.float32), 
                 initialize_initial_state_method='random',
                 POSITIONAL_SET_POINT=np.array([0.,0.], dtype=np.float32),
                 train_hyrl=False, 
                 Mij_complete=None, 
                 j_train=0, 
                 clip_through_obstacles=True, 
                 train=True, 
                 STATIC_OBSTACLE_LOCATIONS=[],
                 NUMBER_OF_STATIC_OBSTACLES=2, 
                 RANDOM_NUMBER_OF_STATIC_OBSTACLES=False, 
                 STATIC_OBSTACLE_RADII=0.5,
                 margin=0.3, 
                 INITIAL_FOCUSED_OBSTACLE=None, 
                 init_in=True, 
                 trained_model=None,
                 VEHICLE_MODEL='Dubin', 
                 IGNORE_OBSTACLES=False,
                 DYNAMIC_OBSTACLE_LOCATIONS=[],
                 NUMBER_OF_DYNAMIC_OBSTACLES=0,
                 DYNAMIC_OBSTACLE_RADII=0.5,
                 OBSTACLE_MODE='static',
                 USE_IMAGE=True,
                 HORIZONTAL_MOVEMENT_AMPLITUDES=0.,
                 VERTICAL_MOVEMENT_AMPLITUDES=0.5,
                 HORIZONTAL_MOVEMENT_PERIODS=10,
                 VERTICAL_MOVEMENT_PERIODS=10,
                 FUTURE_STEPS_DYNAMIC_OBSTACLE=0,
                 DYNAMIC_OBSTACLE_MOTION='random',
                 NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS=0,
                 NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS=1,
                 USE_TIMER=False,
                 INITIAL_TIME_ELAPSED=0,
                 EVOLVE_TIME=False,
                 setpoint_radius=0.1):

        self.TIME_STEPS = TIME_STEPS
        self.initialize_state_randomly = initialize_state_randomly
        self.INITIAL_STATE = INITIAL_STATE
        self.initialize_initial_state_method = initialize_initial_state_method
        self.clip_through_obstacles = clip_through_obstacles # allows agent to clip through STATIC_OBSTACLE_LOCATIONS
        self.train = train
        self.RANDOM_NUMBER_OF_STATIC_OBSTACLES = RANDOM_NUMBER_OF_STATIC_OBSTACLES
        self.train_hyrl = train_hyrl
        self.SAMPLING_TIME_SECONDS = SAMPLING_TIME_SECONDS
        self.OBSTACLE_MODE = OBSTACLE_MODE
        self.USE_IMAGE = USE_IMAGE
        self.USE_TIMER = USE_TIMER
        self.INITIAL_TIME_ELAPSED = INITIAL_TIME_ELAPSED
        self.EVOLVE_TIME = EVOLVE_TIME
        self.setpoint_radius = setpoint_radius


        self.Mij_complete = Mij_complete
        self.j_train = j_train 
        self.margin = margin
        self.init_in = init_in
        self.trained_model = trained_model
        self.VEHICLE_MODEL = VEHICLE_MODEL
        self.IGNORE_OBSTACLES = IGNORE_OBSTACLES

        self.MINIMAL_VALUE_POSITION_X, self.MAXIMAL_VALUE_POSITION_X = -10., 10.
        self.MINIMAL_VALUE_POSITION_Y, self.MAXIMAL_VALUE_POSITION_Y = -10., 10.
        self.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY = 1
        self.MAGNITUDE_CONTROL_INPUT_ORIENTATION = 1 
        self.RESOLUTION_BIRDSEYEVIEW_IMAGE = 64 
        self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE = 2 
        
        self.HORIZONTAL_POSITIONAL_SET_POINT, self.VERTICAL_POSITIONAL_SET_POINT = POSITIONAL_SET_POINT[0], POSITIONAL_SET_POINT[1]
        self.STATIC_OBSTACLE_LOCATIONS = STATIC_OBSTACLE_LOCATIONS
        self.DYNAMIC_OBSTACLE_LOCATIONS = DYNAMIC_OBSTACLE_LOCATIONS
        self.NUMBER_OF_OBSERVATION_IMAGES = 1
        self.FUTURE_STEPS_DYNAMIC_OBSTACLE = FUTURE_STEPS_DYNAMIC_OBSTACLE
        self.DYNAMIC_OBSTACLE_MOTION = DYNAMIC_OBSTACLE_MOTION # 'random', 'deterministic'
        if self.DYNAMIC_OBSTACLE_MOTION == 'deterministic':
            self.COLLECTION_NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS = NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS
            self.COLLECTION_NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS = NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS
            self.COLLECTION_HORIZONTAL_MOVEMENT_AMPLITUDES = HORIZONTAL_MOVEMENT_AMPLITUDES
            self.COLLECTION_VERTICAL_MOVEMENT_AMPLITUDES = VERTICAL_MOVEMENT_AMPLITUDES
            self.COLLECTION_HORIZONTAL_MOVEMENT_PERIODS = HORIZONTAL_MOVEMENT_PERIODS
            self.COLLECTION_VERTICAL_MOVEMENT_PERIODS = VERTICAL_MOVEMENT_PERIODS
        elif self.DYNAMIC_OBSTACLE_MOTION == 'random':
            self.COLLECTION_NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS = []
            self.COLLECTION_NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS = []
            self.COLLECTION_HORIZONTAL_MOVEMENT_AMPLITUDES = []
            self.COLLECTION_VERTICAL_MOVEMENT_AMPLITUDES = []
            self.COLLECTION_HORIZONTAL_MOVEMENT_PERIODS = []
            self.COLLECTION_VERTICAL_MOVEMENT_PERIODS = []
            for obstacle_index in range(NUMBER_OF_DYNAMIC_OBSTACLES): 
                NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS = np.random.randint(0, 11)
                NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS = np.random.randint(0, 11)
                HORIZONTAL_MOVEMENT_AMPLITUDES = []
                VERTICAL_MOVEMENT_AMPLITUDES = []
                HORIZONTAL_MOVEMENT_PERIODS = []
                VERTICAL_MOVEMENT_PERIODS = []
                for term in range(NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS):
                    HORIZONTAL_MOVEMENT_AMPLITUDES.append(np.random.uniform(-1, 1))
                    HORIZONTAL_MOVEMENT_PERIODS.append(np.random.uniform(5,15))
                for term in range(NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS):
                    VERTICAL_MOVEMENT_AMPLITUDES.append(np.random.uniform(-1, 1))
                    VERTICAL_MOVEMENT_PERIODS.append(np.random.uniform(5,15))
                self.COLLECTION_NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS.append(NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS)
                self.COLLECTION_NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS.append(NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS)
                self.COLLECTION_HORIZONTAL_MOVEMENT_AMPLITUDES.append(HORIZONTAL_MOVEMENT_AMPLITUDES)
                self.COLLECTION_VERTICAL_MOVEMENT_AMPLITUDES.append(VERTICAL_MOVEMENT_AMPLITUDES)
                self.COLLECTION_HORIZONTAL_MOVEMENT_PERIODS.append(HORIZONTAL_MOVEMENT_PERIODS)
                self.COLLECTION_VERTICAL_MOVEMENT_PERIODS.append(VERTICAL_MOVEMENT_PERIODS)

        self.INITIAL_FOCUSED_OBSTACLE = INITIAL_FOCUSED_OBSTACLE
        if self.OBSTACLE_MODE == 'static' or self.OBSTACLE_MODE == 'both':
            self.NUMBER_OF_STATIC_OBSTACLES = min(NUMBER_OF_STATIC_OBSTACLES, 12)
            if self.Mij_complete is not None:
                self.STATIC_OBSTACLE_LOCATIONS = self.Mij_complete.vor.points
            if type(STATIC_OBSTACLE_RADII) == list:
                self.STATIC_OBSTACLE_RADII = STATIC_OBSTACLE_RADII
            else:
                self.STATIC_OBSTACLE_RADII = [STATIC_OBSTACLE_RADII] * (self.NUMBER_OF_STATIC_OBSTACLES+1)

        if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
            self.NUMBER_OF_DYNAMIC_OBSTACLES = min(NUMBER_OF_DYNAMIC_OBSTACLES, 2)
            if type(DYNAMIC_OBSTACLE_RADII) == list:
                self.DYNAMIC_OBSTACLE_RADII = DYNAMIC_OBSTACLE_RADII
            else:
                self.DYNAMIC_OBSTACLE_RADII = [DYNAMIC_OBSTACLE_RADII] * (self.NUMBER_OF_DYNAMIC_OBSTACLES+1)
            self.NUMBER_OF_OBSERVATION_IMAGES = 2
            self.INITIAL_DYNAMIC_OBSTACLE_LOCATIONS = None
        else:
            self.NUMBER_OF_DYNAMIC_OBSTACLES = 0

        if self.VEHICLE_MODEL == 'Dubin':
            low_state = [-12, -np.pi]
            high_state = [12,  np.pi]
            for _ in range(self.NUMBER_OF_DYNAMIC_OBSTACLES):
                for _ in range(self.FUTURE_STEPS_DYNAMIC_OBSTACLE+1):
                    low_state.append(-12)
                    low_state.append(-np.pi)
                    high_state.append(12)
                    high_state.append(np.pi)
            if self.USE_TIMER == True:
                low_state.append(0)
                high_state.append(self.SAMPLING_TIME_SECONDS*self.TIME_STEPS)
            
            self.low_state = np.array(low_state, dtype=np.float32)
            self.high_state = np.array(high_state, dtype=np.float32)
            LENGTH_OBSERVATION_VECTOR = len(low_state)

        elif self.VEHICLE_MODEL == 'pointmass':
            self.low_state = np.array([self.MINIMAL_VALUE_POSITION_X,
                                    self.MINIMAL_VALUE_POSITION_Y],
                                    dtype=np.float32)
            self.high_state = np.array([self.MAXIMAL_VALUE_POSITION_X,
                                        self.MAXIMAL_VALUE_POSITION_Y],
                                    dtype=np.float32)
            LENGTH_OBSERVATION_VECTOR = 2


        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32
        )
        if self.USE_IMAGE == True:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(
                        low=0,
                        high=1,
                        # we add one dimension here to act as a "channel" 
                        shape=(self.NUMBER_OF_OBSERVATION_IMAGES, self.RESOLUTION_BIRDSEYEVIEW_IMAGE, self.RESOLUTION_BIRDSEYEVIEW_IMAGE),
                        dtype=np.uint8
                        ),
                "vector": spaces.Box(
                    low=self.low_state,
                    high=self.high_state,
                    shape=(LENGTH_OBSERVATION_VECTOR,),
                    dtype=np.float32
                    )
            })
        else:
            self.observation_space = spaces.Dict({
                "vector": spaces.Box(
                    low=self.low_state,
                    high=self.high_state,
                    shape=(LENGTH_OBSERVATION_VECTOR,),
                    dtype=np.float32
                    )
            })



        # self.reset() 
    
    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = origin[0], origin[1]
        px, py = point[0], point[1]

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy
    
    def get_image_circle(self, horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle):
        orientation_vehicle_angle = np.arctan2(orientation_vector_vehicle[1], orientation_vector_vehicle[0])
        horizontal_values_image = np.linspace(horizontal_position_vehicle-self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, 
                            horizontal_position_vehicle+self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, self.RESOLUTION_BIRDSEYEVIEW_IMAGE)
        vertical_values_image = np.linspace(vertical_position_vehicle-self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, 
                            vertical_position_vehicle+self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, self.RESOLUTION_BIRDSEYEVIEW_IMAGE)
        horizontal_meshgrid_image, vertical_meshgrid_image = np.meshgrid(horizontal_values_image,vertical_values_image)
        image = np.zeros((self.RESOLUTION_BIRDSEYEVIEW_IMAGE, self.RESOLUTION_BIRDSEYEVIEW_IMAGE))

        if self.OBSTACLE_MODE == 'static' or self.OBSTACLE_MODE == 'both':
            for index_of_obstacle, obstacle_location in enumerate(self.STATIC_OBSTACLE_LOCATIONS):
                horizontal_position_obstacle, vertical_position_obstacle = obstacle_location[0], obstacle_location[1]
                rotated_horizontal_values_image, rotated_vertical_values_image = self.rotate((horizontal_position_vehicle, vertical_position_vehicle), (horizontal_position_obstacle, vertical_position_obstacle), 
                                        -orientation_vehicle_angle)
                image += (horizontal_meshgrid_image-rotated_horizontal_values_image)**2 + (vertical_meshgrid_image-rotated_vertical_values_image)**2 < self.STATIC_OBSTACLE_RADII[index_of_obstacle]**2
        if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
            for index_of_obstacle, obstacle_location in enumerate(self.DYNAMIC_OBSTACLE_LOCATIONS):
                horizontal_position_obstacle, vertical_position_obstacle = obstacle_location[0], obstacle_location[1]
                rotated_horizontal_values_image, rotated_vertical_values_image = self.rotate((horizontal_position_vehicle, vertical_position_vehicle), (horizontal_position_obstacle, vertical_position_obstacle), 
                                        -orientation_vehicle_angle)
                image += (horizontal_meshgrid_image-rotated_horizontal_values_image)**2 + (vertical_meshgrid_image-rotated_vertical_values_image)**2 < self.DYNAMIC_OBSTACLE_RADII[index_of_obstacle]**2
        image = np.clip(image, 0, 1).astype(np.uint8)
        if self.IGNORE_OBSTACLES:
            image = np.clip(image, 0, 0).astype(np.uint8)

        return image.reshape((1,self.RESOLUTION_BIRDSEYEVIEW_IMAGE, self.RESOLUTION_BIRDSEYEVIEW_IMAGE))

    def get_distance_obs(self, obst, obstacle_radius, horizontal_position_vehicle, vertical_position_vehicle):
        return max(np.sqrt((horizontal_position_vehicle-obst[0])**2 + \
                            (vertical_position_vehicle-obst[1])**2) - obstacle_radius, 0.)
        
    def update_observation(self, horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle=np.array([1, 0], dtype=np.float32)):  
        if self.VEHICLE_MODEL == 'Dubin':
            desired_orientation = np.arctan2(self.VERTICAL_POSITIONAL_SET_POINT-vertical_position_vehicle, self.HORIZONTAL_POSITIONAL_SET_POINT-horizontal_position_vehicle)
            vehicle_orientation_angle = np.arctan2(orientation_vector_vehicle[1], orientation_vector_vehicle[0])
            orientation_error = desired_orientation-vehicle_orientation_angle
            orientation_error = (orientation_error + np.pi) % (2*np.pi) - np.pi
            distance_error = np.sqrt((self.HORIZONTAL_POSITIONAL_SET_POINT-horizontal_position_vehicle)**2 + (self.VERTICAL_POSITIONAL_SET_POINT-vertical_position_vehicle)**2)
            if self.USE_IMAGE == True:
                new_image = self.get_image_circle(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
                images = self.state['image']
                if self.NUMBER_OF_OBSERVATION_IMAGES > 1:
                    
                    for image_index in range(self.NUMBER_OF_OBSERVATION_IMAGES-1, 0, -1):
                        images[image_index] = images[image_index-1]
                images[0] = new_image[0]
            observation_list = [distance_error, orientation_error]
            if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
                for index_of_obstacle, dynamic_obstacle_location in enumerate(self.DYNAMIC_OBSTACLE_LOCATIONS):
                    for future_steps in range(self.FUTURE_STEPS_DYNAMIC_OBSTACLE+1):
                        future_dynamic_obstacle_location = self.evolve_dynamic_obstacle(dynamic_obstacle_location, index_of_obstacle, future_steps)
                        distance_to_dynamic_obstacle = np.sqrt((future_dynamic_obstacle_location[0]-horizontal_position_vehicle)**2 + (future_dynamic_obstacle_location[1]-vertical_position_vehicle)**2)
                        relative_orientation_to_dynamic_obstacle = np.arctan2(future_dynamic_obstacle_location[1]-vertical_position_vehicle, future_dynamic_obstacle_location[0]-horizontal_position_vehicle)
                        orientation_error_to_dynamic_obstacle = relative_orientation_to_dynamic_obstacle-vehicle_orientation_angle
                        orientation_error_to_dynamic_obstacle = (orientation_error_to_dynamic_obstacle + np.pi) % (2*np.pi) - np.pi
                        observation_list.append(distance_to_dynamic_obstacle)
                        observation_list.append(orientation_error_to_dynamic_obstacle)
                if self.USE_TIMER == True:
                    observation_list.append(self.time_elapsed)

            observation_vector = np.array(observation_list, dtype=np.float32)
        elif self.VEHICLE_MODEL == 'pointmass':
            # no rotation for the image
            if self.USE_IMAGE == True:
                images = self.get_image_circle(horizontal_position_vehicle, vertical_position_vehicle, np.array([1, 0], dtype=np.float32))
            observation_vector = np.array([self.HORIZONTAL_POSITIONAL_SET_POINT-horizontal_position_vehicle,
                                     self.VERTICAL_POSITIONAL_SET_POINT-vertical_position_vehicle],
                                    dtype=np.float32)
        if self.USE_IMAGE == True:
            state = {'image' : images, 
                    'vector': observation_vector}
        else:
            state = {'vector': observation_vector}
        return state
    
    def get_mindist_to_obs(self, horizontal_position_vehicle, vertical_position_vehicle):
        distance_to_closest_obstacle = 100
        if self.OBSTACLE_MODE == 'static' or self.OBSTACLE_MODE == 'both':
            for index_of_obstacle, obstacle_location in enumerate(self.STATIC_OBSTACLE_LOCATIONS):
                d_obi = self.get_distance_obs(obstacle_location, self.STATIC_OBSTACLE_RADII[index_of_obstacle], horizontal_position_vehicle, vertical_position_vehicle)
                if d_obi < distance_to_closest_obstacle:
                    distance_to_closest_obstacle = d_obi
                    closest_id = index_of_obstacle
        if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
            for index_of_obstacle, obstacle_location in enumerate(self.DYNAMIC_OBSTACLE_LOCATIONS):
                d_obi = self.get_distance_obs(obstacle_location, self.DYNAMIC_OBSTACLE_RADII[index_of_obstacle], horizontal_position_vehicle, vertical_position_vehicle)
                if d_obi < distance_to_closest_obstacle:
                    distance_to_closest_obstacle = d_obi
                    closest_id = index_of_obstacle
        return distance_to_closest_obstacle, closest_id

    def check_bounds(self, new_horizontal_position_vehicle, new_vertical_position_vehicle, new_orientation_vector_vehicle=np.array([1,0],dtype=np.float32)):
        distance_to_closest_obstacle, _ = self.get_mindist_to_obs(new_horizontal_position_vehicle, new_vertical_position_vehicle)
        collision_with_obstacle = distance_to_closest_obstacle <= 0
        
        self.out_of_bounds = collision_with_obstacle or \
                        abs(self.vertical_position_vehicle) >= self.MAXIMAL_VALUE_POSITION_Y or \
                            self.horizontal_position_vehicle > self.MAXIMAL_VALUE_POSITION_X or \
                                self.horizontal_position_vehicle < self.MINIMAL_VALUE_POSITION_X

        if self.train_hyrl:
            if self.OBSTACLE_MODE == 'static':
                conditions_for_being_close_to_obstacle = np.zeros(len(self.STATIC_OBSTACLE_LOCATIONS))
                for index_obstacle, obstacle_location in enumerate(self.STATIC_OBSTACLE_LOCATIONS):
                    horizontal_position_obstacle, vertical_position_obstacle = obstacle_location[0], obstacle_location[1]
                    conditions_for_being_close_to_obstacle[index_obstacle] = (horizontal_position_obstacle-self.horizontal_position_vehicle)**2 +\
                        (vertical_position_obstacle-self.vertical_position_vehicle)**2 <  (self.STATIC_OBSTACLE_RADII[index_obstacle]+self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2)**2
                in_Mij = np.zeros(len(self.STATIC_OBSTACLE_LOCATIONS))
                for idx_lambda in range(self.NUMBER_OF_STATIC_OBSTACLES):
                    in_Mij[idx_lambda] = self.Mij_complete.get_in_Mi((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                    self.j_train, idx_lambda)
                self.out_of_bounds = self.out_of_bounds or (np.sum(in_Mij) == 0 and np.sum(conditions_for_being_close_to_obstacle) >= 1)
            if self.OBSTACLE_MODE == 'dynamic': 
                conditions_for_being_close_to_obstacle = np.zeros(len(self.DYNAMIC_OBSTACLE_LOCATIONS))
                for index_obstacle, obstacle_location in enumerate(self.DYNAMIC_OBSTACLE_LOCATIONS):
                    horizontal_position_obstacle, vertical_position_obstacle = obstacle_location[0], obstacle_location[1]
                    conditions_for_being_close_to_obstacle[index_obstacle] = (horizontal_position_obstacle-self.horizontal_position_vehicle)**2 +\
                        (vertical_position_obstacle-self.vertical_position_vehicle)**2 <  (self.DYNAMIC_OBSTACLE_RADII[index_obstacle]+self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2)**2
                in_Mij = np.zeros(len(self.DYNAMIC_OBSTACLE_LOCATIONS))
                for idx, Mijs in enumerate(self.Mij_focused_obstacles):
                    in_Mij[idx] = Mijs.get_in_Mi((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                    self.j_train, new_locations_obstacle=self.DYNAMIC_OBSTACLE_LOCATIONS[self.focused_obstacle], tau=self.time_elapsed)
                self.out_of_bounds = self.out_of_bounds or (np.sum(in_Mij) == 0 and np.sum(conditions_for_being_close_to_obstacle) >= 1)
        if self.out_of_bounds == False or self.clip_through_obstacles == True:
            velocity_magnitude = np.sqrt(((new_horizontal_position_vehicle-self.horizontal_position_vehicle)/self.SAMPLING_TIME_SECONDS)**2 + ((new_vertical_position_vehicle-self.vertical_position_vehicle)/self.SAMPLING_TIME_SECONDS)**2)
            self.horizontal_position_vehicle = new_horizontal_position_vehicle
            self.vertical_position_vehicle = new_vertical_position_vehicle
            if velocity_magnitude > 0.05:
                self.orientation_vector_vehicle = new_orientation_vector_vehicle/np.linalg.norm(new_orientation_vector_vehicle)
                
    def get_reward(self, state):        
        observation_vector = state['vector']
        if self.VEHICLE_MODEL == 'pointmass':
            horizontal_position_error = observation_vector[0]
            vertical_position_error = observation_vector[1]
            distance_to_positional_set_point = np.sqrt(horizontal_position_error**2+vertical_position_error**2)
        else:
            distance_to_positional_set_point = observation_vector[0]

        distance_to_closest_obstacle, index_of_obstacle = self.get_mindist_to_obs(self.horizontal_position_vehicle, self.vertical_position_vehicle)
        if self.train_hyrl:
            if self.OBSTACLE_MODE == 'static':
                boundary_hyrl_kwargs = dict(point=(self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                j=self.j_train, 
                                                WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE=self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE)
            if self.OBSTACLE_MODE == 'dynamic':
                boundary_hyrl_kwargs = dict(point=(self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                            j=self.j_train, 
                                            WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE=self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE,
                                            new_locations_obstacle=self.DYNAMIC_OBSTACLE_LOCATIONS[self.focused_obstacle],) # TODO extend to static + dynamic

            if self.USE_TIMER:
                boundary_hyrl_kwargs['tau'] = self.time_elapsed
            
            distance_to_boundary_hyrl_sets = self.Mij_complete.distance_to_boundary_hyrl((self.horizontal_position_vehicle, self.vertical_position_vehicle), self.j_train, self.focused_obstacle)
            distance_to_closest_obstacle = min(distance_to_closest_obstacle, distance_to_boundary_hyrl_sets)

        if distance_to_closest_obstacle < self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2:
            distance_to_closest_obstacle = max(1e-6, distance_to_closest_obstacle)
            barrier_function = (distance_to_closest_obstacle-0.5*self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE)**2*np.log(0.5*self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/distance_to_closest_obstacle)
        else:
            barrier_function = 0

        if self.IGNORE_OBSTACLES:
            BARRIER_PENALTY = 0
        else:
            BARRIER_PENALTY = 5

        reward_time_step = -distance_to_positional_set_point*self.SAMPLING_TIME_SECONDS - BARRIER_PENALTY*barrier_function*self.SAMPLING_TIME_SECONDS 
        if self.VEHICLE_MODEL=='Dubin':
            orientation_error = observation_vector[1]
            penalty_orientation = min(distance_to_positional_set_point, 1)
            reward_time_step -= 1/np.pi*abs(orientation_error)*penalty_orientation*self.SAMPLING_TIME_SECONDS


        if self.train_hyrl:
            if self.OBSTACLE_MODE == 'static':
                in_Mij = np.zeros(len(self.STATIC_OBSTACLE_LOCATIONS))
                for idx_lambda in range(self.NUMBER_OF_STATIC_OBSTACLES):
                    in_Mij[idx_lambda] = self.Mij_complete.get_in_Mi((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                    self.j_train, idx_lambda)
                if sum(in_Mij) == 0:
                    # distance_to_closest_obstacle = 0
                    reward_time_step -= 1*self.SAMPLING_TIME_SECONDS
            if self.OBSTACLE_MODE == 'dynamic':
                in_Mij = np.zeros(len(self.DYNAMIC_OBSTACLE_LOCATIONS))
                for idx, Mijs in range(self.NUMBER_OF_STATIC_OBSTACLES):
                    in_Mij[idx] = Mijs.get_in_Mi((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                    self.j_train, new_locations_obstacle=self.DYNAMIC_OBSTACLE_LOCATIONS[self.focused_obstacle], tau=self.time_elapsed)
                if sum(in_Mij) == 0:
                    # distance_to_closest_obstacle = 0
                    reward_time_step -= 1*self.SAMPLING_TIME_SECONDS


        return reward_time_step
    
    def check_set_point(self, state):
        observation_vector = state['vector']
        horizontal_position_error = observation_vector[0]
        vertical_position_error = observation_vector[1]
        distance_to_positional_set_point = np.sqrt(horizontal_position_error**2+vertical_position_error**2)
        if distance_to_positional_set_point <= self.setpoint_radius:
            reached_set_point = True
        else:
            reached_set_point = False
        return reached_set_point
    
    def evolve_dynamic_obstacle(self, dynamic_obstacle_location, dynamic_obstacle_index, future_steps):
        time_point = self.time_elapsed + future_steps*self.SAMPLING_TIME_SECONDS
        INITIAL_DYNAMIC_OBSTACLE_LOCATION = self.INITIAL_DYNAMIC_OBSTACLE_LOCATIONS[dynamic_obstacle_index]
        NEW_HORIZONTAL_POSITION_DYNAMIC_OBSTACLE = INITIAL_DYNAMIC_OBSTACLE_LOCATION[0]
        NEW_VERTICAL_POSITION_DYNAMIC_OBSTACLE = INITIAL_DYNAMIC_OBSTACLE_LOCATION[1]
        for term in range(self.COLLECTION_NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS[dynamic_obstacle_index]):
            NEW_HORIZONTAL_POSITION_DYNAMIC_OBSTACLE += self.COLLECTION_HORIZONTAL_MOVEMENT_AMPLITUDES[dynamic_obstacle_index][term]*np.sin(time_point*2*np.pi/self.COLLECTION_HORIZONTAL_MOVEMENT_PERIODS[dynamic_obstacle_index][term])
        for term in range(self.COLLECTION_NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS[dynamic_obstacle_index]):
            NEW_VERTICAL_POSITION_DYNAMIC_OBSTACLE += self.COLLECTION_VERTICAL_MOVEMENT_AMPLITUDES[dynamic_obstacle_index][term]*np.sin(time_point*2*np.pi/self.COLLECTION_VERTICAL_MOVEMENT_PERIODS[dynamic_obstacle_index][term])
        NEW_DYNAMIC_OBSTACLE_LOCATION = (NEW_HORIZONTAL_POSITION_DYNAMIC_OBSTACLE, NEW_VERTICAL_POSITION_DYNAMIC_OBSTACLE)
        return NEW_DYNAMIC_OBSTACLE_LOCATION
    
    def dynamics_vehicle(self, time, vehicle_state, action):
        if self.VEHICLE_MODEL == 'Dubin':
            orientation_vector_vehicle = np.array(vehicle_state[2:], dtype=np.float32)
            control_input_forward_velocity  = action[0]*self.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY
            control_input_orientation       = action[1]*self.MAGNITUDE_CONTROL_INPUT_ORIENTATION
            timederivative_orientation_vector_vehicle  = control_input_orientation*np.array([orientation_vector_vehicle[1], -orientation_vector_vehicle[0]])
            timederivative_horizontal_position_vehicle = control_input_forward_velocity*orientation_vector_vehicle[0]
            timederivative_vertical_position_vehicle   = control_input_forward_velocity*orientation_vector_vehicle[1]
            timederivative_vehicle_state = [timederivative_horizontal_position_vehicle, timederivative_vertical_position_vehicle] + list(timederivative_orientation_vector_vehicle)
        elif self.VEHICLE_MODEL == 'pointmass':
            control_input_horizontal_velocity = action[0]*self.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY
            control_input_vertical_velocity = action[1]*self.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY
            timederivative_horizontal_position_vehicle = control_input_horizontal_velocity
            timederivative_vertical_position_vehicle = control_input_vertical_velocity
            timederivative_vehicle_state = [timederivative_horizontal_position_vehicle, timederivative_vertical_position_vehicle]
        return timederivative_vehicle_state

    def step(self, action):  
        if self.EVOLVE_TIME:
            self.time_elapsed += self.SAMPLING_TIME_SECONDS     
        # updating horizontal_position_vehicle and vertical_position_vehicle 
        if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
            NEW_DYNAMIC_OBSTACLE_LOCATIONS = []
            INITIAL_DYNAMIC_OBSTACLE_LOCATIONS = self.INITIAL_DYNAMIC_OBSTACLE_LOCATIONS.copy()
            for index_of_obstacle, _ in enumerate(self.DYNAMIC_OBSTACLE_LOCATIONS):
                NEW_DYNAMIC_OBSTACLE_LOCATION = self.evolve_dynamic_obstacle(_, index_of_obstacle, 0)
                NEW_DYNAMIC_OBSTACLE_LOCATIONS.append(NEW_DYNAMIC_OBSTACLE_LOCATION)
            self.DYNAMIC_OBSTACLE_LOCATIONS = NEW_DYNAMIC_OBSTACLE_LOCATIONS
        
        vehicle_state = [self.horizontal_position_vehicle, self.vertical_position_vehicle] + list(self.orientation_vector_vehicle)
        timederivative_vehicle_state = self.dynamics_vehicle(0, vehicle_state, action)
        if self.VEHICLE_MODEL == 'Dubin':
            new_orientation_vector_vehicle  = self.orientation_vector_vehicle + np.array(timederivative_vehicle_state[2:])*self.SAMPLING_TIME_SECONDS
            new_horizontal_position_vehicle = self.horizontal_position_vehicle + timederivative_vehicle_state[0]*self.SAMPLING_TIME_SECONDS 
            new_vertical_position_vehicle   = self.vertical_position_vehicle + timederivative_vehicle_state[1]*self.SAMPLING_TIME_SECONDS 
            # Check if illegal states are reached
            self.check_bounds(new_horizontal_position_vehicle, new_vertical_position_vehicle, new_orientation_vector_vehicle)

            self.state = self.update_observation(self.horizontal_position_vehicle, self.vertical_position_vehicle, self.orientation_vector_vehicle)
        
        elif self.VEHICLE_MODEL == 'pointmass':
            new_horizontal_position_vehicle = self.horizontal_position_vehicle + timederivative_vehicle_state[0]*self.SAMPLING_TIME_SECONDS 
            new_vertical_position_vehicle = self.vertical_position_vehicle + timederivative_vehicle_state[1]*self.SAMPLING_TIME_SECONDS 
            # Check if illegal states are reached
            self.check_bounds(new_horizontal_position_vehicle, new_vertical_position_vehicle)

            self.state = self.update_observation(self.horizontal_position_vehicle, self.vertical_position_vehicle)
        
        
        # Calculate reward_time_step      
        reward_time_step = self.get_reward(self.state)

        # Update TIME_STEPS left
        self.TIME_STEPS_left -= 1
        
        # Check if set point is reached:
        reached_set_point = self.check_set_point(self.state)

        # Check if simulation is terminated
        if self.TIME_STEPS_left <= 0 or reached_set_point:
            terminated = True
        else:
            terminated = False 
            
        # Set placeholder for info
        info = {}

        
        return self.state, reward_time_step, terminated, False, info
    
    def get_initialize_state_randomly(self):
        if self.initialize_initial_state_method == 'random':
            self.HORIZONTAL_POSITIONAL_SET_POINT, self.VERTICAL_POSITIONAL_SET_POINT = 0, 0
            
            stop_loop = False
            while stop_loop == False:
                self.horizontal_position_vehicle = np.random.uniform(self.MINIMAL_VALUE_POSITION_X+1, self.MAXIMAL_VALUE_POSITION_X-1)
                self.vertical_position_vehicle = np.random.uniform(self.MINIMAL_VALUE_POSITION_Y+1, self.MAXIMAL_VALUE_POSITION_Y-1)
                if self.OBSTACLE_MODE == 'static':
                    OBSTACLE_LOCATIONS_INITIALIZE = self.STATIC_OBSTACLE_LOCATIONS
                    OBSTACLE_RADII_INITIALIZE = self.STATIC_OBSTACLE_RADII
                elif self.OBSTACLE_MODE == 'dynamic':
                    OBSTACLE_LOCATIONS_INITIALIZE = self.DYNAMIC_OBSTACLE_LOCATIONS
                    OBSTACLE_RADII_INITIALIZE = self.DYNAMIC_OBSTACLE_RADII
                elif self.OBSTACLE_MODE == 'both':
                    OBSTACLE_LOCATIONS_INITIALIZE = self.STATIC_OBSTACLE_LOCATIONS + self.DYNAMIC_OBSTACLE_LOCATIONS
                    OBSTACLE_RADII_INITIALIZE = self.STATIC_OBSTACLE_RADII + self.DYNAMIC_OBSTACLE_RADII
                else: 
                    print('no valid obstacle mode passed')
                conditions_for_being_close_to_obstacle = np.zeros(len(OBSTACLE_LOCATIONS_INITIALIZE))
                conditions_for_being_far_from_obstacle = np.zeros(len(OBSTACLE_LOCATIONS_INITIALIZE))
                conditions_for_being_far_from_positional_set_point = np.zeros(len(OBSTACLE_LOCATIONS_INITIALIZE))
                conditions_for_being_close_to_positional_set_point = np.zeros(len(OBSTACLE_LOCATIONS_INITIALIZE))
                for index_obstacle, obstacle_location in enumerate(OBSTACLE_LOCATIONS_INITIALIZE):
                    horizontal_position_obstacle, vertical_position_obstacle = obstacle_location[0], obstacle_location[1]
                    conditions_for_being_close_to_obstacle[index_obstacle] = (horizontal_position_obstacle-self.horizontal_position_vehicle)**2 +\
                        (vertical_position_obstacle-self.vertical_position_vehicle)**2 >  (OBSTACLE_RADII_INITIALIZE[index_obstacle]*1.1)**2 # 2 -> 1.5 -> 1.1
                    conditions_for_being_far_from_positional_set_point[index_obstacle] = (self.horizontal_position_vehicle-self.HORIZONTAL_POSITIONAL_SET_POINT)**2 + \
                        (self.vertical_position_vehicle-self.VERTICAL_POSITIONAL_SET_POINT)**2 >  (OBSTACLE_RADII_INITIALIZE[index_obstacle]*2)**2 # 4 --> 2
                    conditions_for_being_far_from_obstacle[index_obstacle] = (horizontal_position_obstacle-self.horizontal_position_vehicle)**2 +\
                        (vertical_position_obstacle-self.vertical_position_vehicle)**2 <  (2*self.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE)**2# (OBSTACLE_RADII_INITIALIZE[index_obstacle]*18)**2 # 5 -> 2 -> 18 -> 5
                    conditions_for_being_close_to_positional_set_point[index_obstacle] = (self.horizontal_position_vehicle-self.HORIZONTAL_POSITIONAL_SET_POINT)**2 + \
                        (self.vertical_position_vehicle-self.VERTICAL_POSITIONAL_SET_POINT)**2 <  (6)**2
                if self.train_hyrl:
                    if self.OBSTACLE_MODE == 'static':
                        in_Mij = np.zeros(len(self.STATIC_OBSTACLE_LOCATIONS))
                        for idx_lambda in range(self.NUMBER_OF_STATIC_OBSTACLES):
                            if self.init_in:
                                in_Mij[idx_lambda] = self.Mij_complete.get_in_Mi((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                            self.j_train, idx_lambda)
                            else:
                                in_Mij[idx_lambda] = self.Mij_complete.near_boundary((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                            self.j_train, index_lambda=self.focused_obstacle,
                                                            boundary_margin=self.margin)  
                        if np.prod(conditions_for_being_close_to_obstacle)>0 and np.prod(conditions_for_being_far_from_positional_set_point)>0 \
                            and np.sum(conditions_for_being_far_from_obstacle)>=1 and np.sum(in_Mij)>=1 and \
                            np.prod(conditions_for_being_close_to_positional_set_point)>0:
                            stop_loop = True
                    if self.OBSTACLE_MODE == 'dynamic':
                        in_Mij = np.zeros(len(self.DYNAMIC_OBSTACLE_LOCATIONS))
                        for idx, Mijs in enumerate(self.Mij_focused_obstacles):
                            if self.init_in:
                                in_Mij[idx] = Mijs.get_in_Mi((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                            self.j_train, new_locations_obstacle=self.DYNAMIC_OBSTACLE_LOCATIONS[self.focused_obstacle], tau=self.time_elapsed)
                            else:
                                in_Mij[idx] = Mijs.near_boundary((self.horizontal_position_vehicle, self.vertical_position_vehicle), 
                                                            self.j_train, 
                                                            boundary_margin=self.margin, new_locations_obstacle=self.DYNAMIC_OBSTACLE_LOCATIONS[self.focused_obstacle], tau=self.time_elapsed)  
                        if np.prod(conditions_for_being_close_to_obstacle)>0 and np.prod(conditions_for_being_far_from_positional_set_point)>0 \
                            and np.sum(conditions_for_being_far_from_obstacle)>=1 and np.sum(in_Mij)>=1 and \
                            np.prod(conditions_for_being_close_to_positional_set_point)>0:
                            stop_loop = True
                        if self.OBSTACLE_MODE == 'both':
                            Exception('Hyrl for dynamic + static obstacles not embedded yet')
                else:
                    if np.prod(conditions_for_being_close_to_obstacle)>0 and np.prod(conditions_for_being_far_from_positional_set_point)>0 and \
                        np.prod(conditions_for_being_close_to_positional_set_point)>0:
                        stop_loop = True
        
        distance_to_closest_obstacle, _ = self.get_mindist_to_obs(self.horizontal_position_vehicle, self.vertical_position_vehicle)
        collision_with_obstacle = distance_to_closest_obstacle <= 0
        self.out_of_bounds = collision_with_obstacle or \
                       abs(self.vertical_position_vehicle) >= self.MAXIMAL_VALUE_POSITION_Y or \
                           self.horizontal_position_vehicle > self.MAXIMAL_VALUE_POSITION_X or \
                               self.horizontal_position_vehicle < self.MINIMAL_VALUE_POSITION_X
        
    def get_static_obstacle_locations(self):
        # works for 12 STATIC_OBSTACLE_LOCATIONS!
        start_angle = np.random.uniform(0,2*np.pi)
        angle_spread_obs1 = np.linspace(start_angle, 2*np.pi+start_angle, min(4,self.NUMBER_OF_STATIC_OBSTACLES), 
                                        endpoint=False)
        if self.NUMBER_OF_STATIC_OBSTACLES > 4:
            start_angle2 = np.random.uniform(0, 2*np.pi)
            angle_spread_obs2 = np.linspace(start_angle2, 2*np.pi+start_angle2, min(12,self.NUMBER_OF_STATIC_OBSTACLES-4), 
                                            endpoint=False)
        self.STATIC_OBSTACLE_LOCATIONS = []
        for obstacle_number in range(self.NUMBER_OF_STATIC_OBSTACLES) :
            if obstacle_number < 4:
                obstacle_initialization_angle = np.random.uniform(-0.1*np.pi, 0.1*np.pi)
                distance_obs = np.random.uniform(self.STATIC_OBSTACLE_RADII[obstacle_number]*3.5, 
                                                 self.STATIC_OBSTACLE_RADII[obstacle_number]*4.5)
                horizontal_position_obstacle = distance_obs*np.cos(obstacle_initialization_angle+angle_spread_obs1[obstacle_number])
                vertical_position_obstacle = distance_obs*np.sin(obstacle_initialization_angle+angle_spread_obs1[obstacle_number])
            else:
                obstacle_number -= 4
                obstacle_initialization_angle = np.random.uniform(-0.05*np.pi, 0.05*np.pi)
                distance_obs = np.random.uniform(self.STATIC_OBSTACLE_RADII[obstacle_number]*8, 
                                                 self.STATIC_OBSTACLE_RADII[obstacle_number]*10)
                horizontal_position_obstacle = distance_obs*np.cos(obstacle_initialization_angle+angle_spread_obs2[obstacle_number])
                vertical_position_obstacle = distance_obs*np.sin(obstacle_initialization_angle+angle_spread_obs2[obstacle_number])
            self.STATIC_OBSTACLE_LOCATIONS.append((horizontal_position_obstacle, vertical_position_obstacle))

    def get_dynamic_obstacle_locations(self):
        print('Functionality not added yet, define the locations of the moving obstacles')
        print('using locations (-2, 0), (2, 0)')
        standard_locations = [(-2, 0), (2,0)]
        self.DYNAMIC_OBSTACLE_LOCATIONS = standard_locations[:self.NUMBER_OF_DYNAMIC_OBSTACLES]
        print(self.DYNAMIC_OBSTACLE_LOCATIONS)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.time_elapsed = self.INITIAL_TIME_ELAPSED
        if self.train:
            self.time_elapsed = np.random.uniform(0, self.TIME_STEPS*self.SAMPLING_TIME_SECONDS)
        if self.train and not self.train_hyrl:
            if self.OBSTACLE_MODE == 'static' or self.OBSTACLE_MODE == 'both':
                if self.RANDOM_NUMBER_OF_STATIC_OBSTACLES:
                    self.NUMBER_OF_STATIC_OBSTACLES = np.random.randint(1, 12+1)
                if len(self.STATIC_OBSTACLE_LOCATIONS) == 0:
                    self.get_static_obstacle_locations()
            if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
                if len(self.DYNAMIC_OBSTACLE_LOCATIONS) == 0:
                    self.get_dynamic_obstacle_locations()
        if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
            if self.INITIAL_DYNAMIC_OBSTACLE_LOCATIONS is None:
                self.INITIAL_DYNAMIC_OBSTACLE_LOCATIONS = list(self.DYNAMIC_OBSTACLE_LOCATIONS)
        # for hyrl script, initialize at a different time point
        if self.time_elapsed > 0:
            if self.OBSTACLE_MODE == 'dynamic' or self.OBSTACLE_MODE == 'both':
                NEW_DYNAMIC_OBSTACLE_LOCATIONS = []
                for index_of_obstacle, _ in enumerate(self.DYNAMIC_OBSTACLE_LOCATIONS):
                    NEW_DYNAMIC_OBSTACLE_LOCATION = self.evolve_dynamic_obstacle(_, index_of_obstacle, 0)
                    NEW_DYNAMIC_OBSTACLE_LOCATIONS.append(NEW_DYNAMIC_OBSTACLE_LOCATION)
                self.DYNAMIC_OBSTACLE_LOCATIONS = NEW_DYNAMIC_OBSTACLE_LOCATIONS
            
        if self.train_hyrl:
            if self.INITIAL_FOCUSED_OBSTACLE is None:
                if self.OBSTACLE_MODE == 'static':
                    self.focused_obstacle = np.random.randint(self.NUMBER_OF_STATIC_OBSTACLES)
                if self.OBSTACLE_MODE == 'dynamic':
                    self.focused_obstacle = np.random.randint(self.NUMBER_OF_DYNAMIC_OBSTACLES)
                if self.OBSTACLE_MODE == 'both':
                    Exception('Hyrl for dynamic + static obstacles not embedded yet')
                    # we can embed static obstacles as dynamic obstacle that just dont move
            else:
                self.focused_obstacle = self.INITIAL_FOCUSED_OBSTACLE
        self.horizontal_position_vehicle = min(max(self.INITIAL_STATE[0], self.MINIMAL_VALUE_POSITION_X), self.MAXIMAL_VALUE_POSITION_X)
        self.vertical_position_vehicle = min(max(self.INITIAL_STATE[1], self.MINIMAL_VALUE_POSITION_Y), self.MAXIMAL_VALUE_POSITION_Y)
        if self.VEHICLE_MODEL == 'Dubin':
            self.orientation_vector_vehicle = np.array([self.INITIAL_STATE[2], self.INITIAL_STATE[3]])
        if self.initialize_state_randomly==True:
            # initialize system around the initial state
            self.get_initialize_state_randomly()
            while self.out_of_bounds:
                    self.get_initialize_state_randomly()
            if self.VEHICLE_MODEL == 'Dubin':
                orientation_vehicle_angle = np.arctan2(self.VERTICAL_POSITIONAL_SET_POINT-self.vertical_position_vehicle, self.HORIZONTAL_POSITIONAL_SET_POINT-self.horizontal_position_vehicle)
                orientation_vehicle_angle += np.random.uniform(low=-np.pi/2, high=np.pi/2) #* 4 # TODO
                self.orientation_vector_vehicle[0], self.orientation_vector_vehicle[1] = np.cos(orientation_vehicle_angle), np.sin(orientation_vehicle_angle)
        
        if self.VEHICLE_MODEL == 'Dubin':
            if self.USE_IMAGE == True:
                initial_images = np.zeros((self.NUMBER_OF_OBSERVATION_IMAGES, self.RESOLUTION_BIRDSEYEVIEW_IMAGE, self.RESOLUTION_BIRDSEYEVIEW_IMAGE), dtype=np.uint8)
                initial_image = self.get_image_circle(self.horizontal_position_vehicle, self.vertical_position_vehicle, self.orientation_vector_vehicle)
                for image_index in range(self.NUMBER_OF_OBSERVATION_IMAGES):
                    initial_images[image_index] = initial_image[0]
                self.state  = {'image' : initial_images} 
            self.state = self.update_observation(self.horizontal_position_vehicle, self.vertical_position_vehicle, self.orientation_vector_vehicle)
        elif self.VEHICLE_MODEL == 'pointmass':
            self.state = self.update_observation(self.horizontal_position_vehicle, self.vertical_position_vehicle)
        # set the total number of episode TIME_STEPS
        self.TIME_STEPS_left = self.TIME_STEPS
        info = {}
        
        return self.state, info
