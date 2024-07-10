import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from env_static_obstacles import BirdsEyeViewStaticObstacleLocationsEnvironment
from stable_baselines3 import PPO
from hybridequationsolver import  hybridsystem
import imageio
from utils import Def_Mob_general
import pickle


def perturb_observation(env, noise, horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle):
    obs = dict()
    desired_orientation = np.arctan2(env.VERTICAL_POSITIONAL_SET_POINT-(vertical_position_vehicle+noise[1]), 
                                     env.HORIZONTAL_POSITIONAL_SET_POINT-(horizontal_position_vehicle+noise[0]))

    
    noisy_vehicle_orientation_angle = np.arctan2(orientation_vector_vehicle[1], 
                                                 orientation_vector_vehicle[0]) + noise[2]
    noisy_orientation_vector_vehicle = np.array([np.cos(noisy_vehicle_orientation_angle), np.sin(noisy_vehicle_orientation_angle)])
    
    orientation_error = desired_orientation-noisy_vehicle_orientation_angle
    orientation_error = (orientation_error + np.pi) % (2*np.pi) - np.pi
    distance_error = np.sqrt((env.HORIZONTAL_POSITIONAL_SET_POINT-(horizontal_position_vehicle+noise[0]))**2 +\
                            (env.VERTICAL_POSITIONAL_SET_POINT-(vertical_position_vehicle+noise[1]))**2)
    if env.USE_IMAGE == True:
        image = env.get_image_circle((horizontal_position_vehicle+noise[0]), (vertical_position_vehicle+noise[1]), 
                                      noisy_orientation_vector_vehicle)
        obs['image'] = image
    observation_list = [distance_error, orientation_error]
    obs['vector'] = np.array(observation_list, dtype=np.float32)
    return obs

def flow( x, normal_env, hyrlmp_env):
    normal_horizontal_position_vehicle    = x[0]
    normal_vertical_position_vehicle      = x[1]
    normal_orientation_vector_vehicle     = x[2:4]
    normal_control_input_forward_velocity = x[4]
    normal_control_input_orientation      = x[5]
    normal_holding_enemy_flag             = x[6]

    hyrlmp_horizontal_position_vehicle    = x[7]
    hyrlmp_vertical_position_vehicle      = x[8]
    hyrlmp_orientation_vector_vehicle     = x[9:11]
    hyrlmp_control_input_forward_velocity = x[11]
    hyrlmp_control_input_orientation      = x[12]
    hyrlmp_q_logic                        = int(x[13])
    hyrlmp_lambda_logic                   = int(x[14])
    hyrlmp_holding_enemy_flag             = x[15]

    timer                                 = x[16]
    noise_sign                            = x[17]
    normal_points_scored                  = x[18]
    hyrlmp_points_scored                  = x[19]

    normal_xdot = np.zeros(7)
    normal_xdot[0] =  normal_control_input_forward_velocity*normal_env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*normal_orientation_vector_vehicle[0]
    normal_xdot[1] =  normal_control_input_forward_velocity*normal_env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*normal_orientation_vector_vehicle[1]
    normal_xdot[2] =  normal_control_input_orientation*normal_env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*normal_orientation_vector_vehicle[1]
    normal_xdot[3] = -normal_control_input_orientation*normal_env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*normal_orientation_vector_vehicle[0]
    normal_xdot[4] =  0
    normal_xdot[5] =  0
    normal_xdot[6] =  0

    hyrlmp_xdot = np.zeros(9)
    hyrlmp_xdot[0] =  hyrlmp_control_input_forward_velocity*hyrlmp_env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*hyrlmp_orientation_vector_vehicle[0]
    hyrlmp_xdot[1] =  hyrlmp_control_input_forward_velocity*hyrlmp_env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*hyrlmp_orientation_vector_vehicle[1]
    hyrlmp_xdot[2] =  hyrlmp_control_input_orientation*hyrlmp_env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*hyrlmp_orientation_vector_vehicle[1]
    hyrlmp_xdot[3] = -hyrlmp_control_input_orientation*hyrlmp_env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*hyrlmp_orientation_vector_vehicle[0]
    hyrlmp_xdot[4] =  0
    hyrlmp_xdot[5] =  0
    hyrlmp_xdot[6] =  0
    hyrlmp_xdot[7] =  0
    hyrlmp_xdot[8] =  0

    shared_xdot = np.zeros(4)
    shared_xdot[0] = 1
    shared_xdot[1] = 0
    shared_xdot[2] = 0
    shared_xdot[3] = 0

    xdot = np.concatenate((normal_xdot,hyrlmp_xdot,shared_xdot)) 
    return xdot

def jump( x, normal_env, hyrlmp_env, normal_agent, hyrlmp_agent, noise_magnitude_position, noise_magnitude_orientation, HyRL_MP_sets, normal_base_position, hyrlmp_base_position, collision_radius, capture_radius):
    normal_horizontal_position_vehicle    = x[0]
    normal_vertical_position_vehicle      = x[1]
    normal_orientation_vector_vehicle     = x[2:4]
    normal_control_input_forward_velocity = x[4]
    normal_control_input_orientation      = x[5]
    normal_holding_enemy_flag             = x[6]

    hyrlmp_horizontal_position_vehicle    = x[7]
    hyrlmp_vertical_position_vehicle      = x[8]
    hyrlmp_orientation_vector_vehicle     = x[9:11]
    hyrlmp_control_input_forward_velocity = x[11]
    hyrlmp_control_input_orientation      = x[12]
    hyrlmp_q_logic                        = int(x[13])
    hyrlmp_lambda_logic                   = int(x[14])
    hyrlmp_holding_enemy_flag             = x[15]

    timer                                 = x[16]
    noise_sign                            = x[17]
    normal_points_scored                  = x[18]
    hyrlmp_points_scored                  = x[19]


    normal_jumped_horizontal_position_vehicle = normal_horizontal_position_vehicle
    normal_jumped_vertical_position_vehicle = normal_vertical_position_vehicle
    normal_jumped_orientation_vector_vehicle = normal_orientation_vector_vehicle

    hyrlmp_jumped_horizontal_position_vehicle = hyrlmp_horizontal_position_vehicle
    hyrlmp_jumped_vertical_position_vehicle = hyrlmp_vertical_position_vehicle
    hyrlmp_jumped_orientation_vector_vehicle = hyrlmp_orientation_vector_vehicle

    normal_jumped_holding_enemy_flag = normal_holding_enemy_flag
    hyrlmp_jumped_holding_enemy_flag = hyrlmp_holding_enemy_flag

    normal_jumped_points_scored = normal_points_scored
    hyrlmp_jumped_points_scored = hyrlmp_points_scored

    reset_normal = False
    reset_hyrlmp = False
    distance_between_players = np.linalg.norm(np.array([normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle])-
                                              np.array([hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle]))
    
    if distance_between_players < collision_radius:
        # players have collided
        if normal_jumped_holding_enemy_flag == True:
            reset_normal = True 
            hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT = normal_base_position[0], normal_base_position[1]
        if hyrlmp_jumped_holding_enemy_flag == True:
            reset_hyrlmp = True
            normal_env.HORIZONTAL_POSITIONAL_SET_POINT, normal_env.VERTICAL_POSITIONAL_SET_POINT = hyrlmp_base_position[0], hyrlmp_base_position[1]
        if normal_jumped_holding_enemy_flag == False and hyrlmp_jumped_holding_enemy_flag == False:
            reset_normal = True
            reset_hyrlmp = True
    
    
    distance_to_enemy_base_normal = np.linalg.norm(np.array([normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle])-
                                                   np.array([hyrlmp_base_position[0], hyrlmp_base_position[1]]))
    distance_to_enemy_base_hyrlmp = np.linalg.norm(np.array([hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle])-
                                                   np.array([normal_base_position[0], normal_base_position[1]]))
    close_to_enemy_base_normal = distance_to_enemy_base_normal <= capture_radius
    close_to_enemy_base_hyrlmp = distance_to_enemy_base_hyrlmp <= capture_radius

    distance_to_friendly_base_normal = np.linalg.norm(np.array([normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle])-
                                                   np.array([normal_base_position[0], normal_base_position[1]]))
    distance_to_friendly_base_hyrlmp = np.linalg.norm(np.array([hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle])-
                                                   np.array([hyrlmp_base_position[0], hyrlmp_base_position[1]]))
    close_to_friendly_base_normal = distance_to_friendly_base_normal <= capture_radius
    close_to_friendly_base_hyrlmp = distance_to_friendly_base_hyrlmp <= capture_radius

    # checking if normal agent reached it's target / took the enemy flag / captured the enemy flag / needs to chase the hyrlmp agent
    normal_env.check_bounds(normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle, normal_jumped_orientation_vector_vehicle)

    if reset_normal == True or normal_env.out_of_bounds == True:
        # reset normal agent back to its base and return the enemy flag if applicable
        normal_jumped_holding_enemy_flag = False
        normal_env.HORIZONTAL_POSITIONAL_SET_POINT, normal_env.VERTICAL_POSITIONAL_SET_POINT = hyrlmp_base_position[0], hyrlmp_base_position[1]
        normal_jumped_horizontal_position_vehicle = normal_base_position[0]
        normal_jumped_vertical_position_vehicle = normal_base_position[1] 
        normal_jumped_orientation_angle = np.arctan2(hyrlmp_base_position[1]-normal_jumped_vertical_position_vehicle, hyrlmp_base_position[0]-normal_jumped_horizontal_position_vehicle)
        normal_jumped_orientation_vector_vehicle[0], normal_jumped_orientation_vector_vehicle[1] = np.cos(normal_jumped_orientation_angle), np.sin(normal_jumped_orientation_angle)
    else:
        # check if flag is captured and set point is reached
        if hyrlmp_jumped_holding_enemy_flag == True and reset_hyrlmp == False:
            # The normal agent needs to chase it's flag!
            normal_env.STATIC_OBSTACLE_LOCATIONS[0] = (100, 100) # prevent avoidance of the target
            normal_env.HORIZONTAL_POSITIONAL_SET_POINT, normal_env.VERTICAL_POSITIONAL_SET_POINT = hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle
        elif close_to_enemy_base_normal == True and normal_jumped_holding_enemy_flag == False:
            # The normal agent has picked up the enemy flag!
            # if close_to_enemy_base_normal == True:
            normal_jumped_holding_enemy_flag = True
            # The normal agent needs to return the flag to its base!
            normal_env.HORIZONTAL_POSITIONAL_SET_POINT, normal_env.VERTICAL_POSITIONAL_SET_POINT = normal_base_position[0], normal_base_position[1]
        elif close_to_friendly_base_normal == True and normal_jumped_holding_enemy_flag == True:
            # The normal agent has brought the enemy flag to it's base and has scored a point!
            normal_jumped_holding_enemy_flag = False
            normal_env.HORIZONTAL_POSITIONAL_SET_POINT, normal_env.VERTICAL_POSITIONAL_SET_POINT = hyrlmp_base_position[0], hyrlmp_base_position[1]
            hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT = normal_base_position[0], normal_base_position[1]
            normal_jumped_points_scored = normal_points_scored + 1
    # checking if hyrlmp agent reached it's target / took the enemy flag / captured the enemy flag / needs to chase the normal agent
    hyrlmp_env.check_bounds(hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle, hyrlmp_jumped_orientation_vector_vehicle)
    # hyrlmp_observation = hyrlmp_env.update_observation(hyrlmp_horizontal_position_vehicle, hyrlmp_vertical_position_vehicle, hyrlmp_orientation_vector_vehicle)

    if reset_hyrlmp == True or hyrlmp_env.out_of_bounds == True:
        # reset normal agent back to its base and return the enemy flag if applicable
        hyrlmp_jumped_holding_enemy_flag = False
        hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT = normal_base_position[0], normal_base_position[1]
        hyrlmp_jumped_horizontal_position_vehicle = hyrlmp_base_position[0]
        hyrlmp_jumped_vertical_position_vehicle = hyrlmp_base_position[1]
        hyrlmp_jumped_orientation_angle = np.arctan2(normal_base_position[1]-hyrlmp_jumped_vertical_position_vehicle, normal_base_position[0]-hyrlmp_jumped_horizontal_position_vehicle)
        hyrlmp_jumped_orientation_vector_vehicle[0], hyrlmp_jumped_orientation_vector_vehicle[1] = np.cos(hyrlmp_jumped_orientation_angle), np.sin(hyrlmp_jumped_orientation_angle)
    else:
        # hyrlmp_reached_set_point = hyrlmp_env.check_set_point(hyrlmp_observation)
        if normal_jumped_holding_enemy_flag == True and reset_normal == False:
            # The hyrlmp agent needs to chase it's flag!
            hyrlmp_env.STATIC_OBSTACLE_LOCATIONS[0] = (100, 100) # prevent avoidance of the target
            hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT = normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle
        elif close_to_enemy_base_hyrlmp == True and hyrlmp_jumped_holding_enemy_flag == False:
            # The hyrlmp agent has picked up the enemy flag!
            # if close_to_enemy_base_hyrlmp == True:
            hyrlmp_jumped_holding_enemy_flag = True
            # The hyrlmp agent needs to return the flag to its base!
            hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT = hyrlmp_base_position[0], hyrlmp_base_position[1]
        elif close_to_friendly_base_hyrlmp == True and hyrlmp_jumped_holding_enemy_flag == True:
            # The hyrlmp agent has brought the enemy flag to it's base and has scored a point!
            hyrlmp_jumped_holding_enemy_flag = False
            hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT = normal_base_position[0], normal_base_position[1]
            normal_env.HORIZONTAL_POSITIONAL_SET_POINT, normal_env.VERTICAL_POSITIONAL_SET_POINT = hyrlmp_base_position[0], hyrlmp_base_position[1]
            hyrlmp_jumped_points_scored = hyrlmp_points_scored + 1
    
    # updating dynamic obstacle locations in both environments
    if hyrlmp_jumped_holding_enemy_flag == False:       
        normal_env.STATIC_OBSTACLE_LOCATIONS[0] = (hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle)
    if normal_jumped_holding_enemy_flag == False:
        hyrlmp_env.STATIC_OBSTACLE_LOCATIONS[0] = (normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle)
    HyRL_MP_sets.set_voronoi_partition(hyrlmp_env.STATIC_OBSTACLE_LOCATIONS, hyrlmp_env.STATIC_OBSTACLE_RADII)
    HyRL_MP_sets.POSITIONAL_SET_POINT = np.array([hyrlmp_env.HORIZONTAL_POSITIONAL_SET_POINT, hyrlmp_env.VERTICAL_POSITIONAL_SET_POINT])

    noise = np.array([noise_sign*noise_magnitude_position, -noise_sign*noise_magnitude_position, noise_sign*noise_magnitude_orientation])
    
    hyrlmp_perturbed_vehicle_position = np.array([hyrlmp_jumped_horizontal_position_vehicle+noise[0], hyrlmp_jumped_vertical_position_vehicle+noise[1]])
   
    hyrlmp_jumped_q_logic, hyrlmp_jumped_lambda_logic = jump_HyRL_MP_logic_variables(HyRL_MP_sets, hyrlmp_perturbed_vehicle_position, hyrlmp_q_logic, hyrlmp_lambda_logic)
    hyrlmp_agent = hyrlmp_agent[hyrlmp_jumped_q_logic]
    
    normal_noisy_observation = perturb_observation(normal_env, noise, normal_jumped_horizontal_position_vehicle, normal_jumped_vertical_position_vehicle, normal_jumped_orientation_vector_vehicle)
    hyrlmp_noisy_observation = perturb_observation(hyrlmp_env, noise, hyrlmp_jumped_horizontal_position_vehicle, hyrlmp_jumped_vertical_position_vehicle, hyrlmp_jumped_orientation_vector_vehicle)
   
    normal_jumped_action, _ = normal_agent.predict(normal_noisy_observation, deterministic=True)
    normal_jumped_control_input_forward_velocity = normal_jumped_action[0]
    normal_jumped_control_input_orientation      = normal_jumped_action[1]

    hyrlmp_jumped_action, _ = hyrlmp_agent.predict(hyrlmp_noisy_observation, deterministic=True)
    hyrlmp_jumped_control_input_forward_velocity = hyrlmp_jumped_action[0]
    hyrlmp_jumped_control_input_orientation      = hyrlmp_jumped_action[1]
    

    jumped_timer                          = 0
    jumped_noise_sign                     = -noise_sign

    jumped_x = [normal_jumped_horizontal_position_vehicle,   
                normal_jumped_vertical_position_vehicle,  
                normal_jumped_orientation_vector_vehicle[0], 
                normal_jumped_orientation_vector_vehicle[1],   
                normal_jumped_control_input_forward_velocity, 
                normal_jumped_control_input_orientation,     
                normal_jumped_holding_enemy_flag,  
                hyrlmp_jumped_horizontal_position_vehicle,  
                hyrlmp_jumped_vertical_position_vehicle,     
                hyrlmp_jumped_orientation_vector_vehicle[0],    
                hyrlmp_jumped_orientation_vector_vehicle[1],   
                hyrlmp_jumped_control_input_forward_velocity, 
                hyrlmp_jumped_control_input_orientation,      
                hyrlmp_jumped_q_logic,                        
                hyrlmp_jumped_lambda_logic,                  
                hyrlmp_jumped_holding_enemy_flag,            
                jumped_timer,                                 
                jumped_noise_sign,                           
                normal_jumped_points_scored,                 
                hyrlmp_jumped_points_scored]
    return jumped_x

def inside_C(x, normal_env, hyrlmp_env,):
    timer = x[16]

    if (timer <= normal_env.SAMPLING_TIME_SECONDS):
        inside = 1
    else:
        inside = 0
    return inside

def inside_D(x, normal_env, hyrlmp_env,):
    timer = x[16]

    if (timer >= normal_env.SAMPLING_TIME_SECONDS):
        inside = 1
    else:
        inside = 0
    return inside

def jump_HyRL_MP_logic_variables(HyRL_MP_sets, vehicle_position, q_logic, lambda_logic):
    # checking if we're in the voronoi partition of the focused obstacle
    in_D_hyst_qs_lambda = HyRL_MP_sets.check_voronoi(vehicle_position, lambda_logic, res=20)
    if in_D_hyst_qs_lambda == True:
       in_D_hyst_q_lambda = HyRL_MP_sets.check_in_Mi(vehicle_position, q_logic, lambda_logic)
       jumped_lambda_logic = lambda_logic
       if in_D_hyst_q_lambda == True:
           jumped_q_logic = q_logic
       else:
           jumped_q_logic = int(abs(1-q_logic))
    else:
        # we need to change our focused obstacle and possibly the value of q
        potential_q_lambda_pairs = []
        for potential_q_logic in [0, 1]:
            for potential_lambda_logic in range(len(HyRL_MP_sets.vor.points)):
                if HyRL_MP_sets.get_in_Mi(vehicle_position, potential_q_logic, potential_lambda_logic) == True:
                    potential_q_lambda_pairs.append((potential_q_logic, potential_lambda_logic))
        jumped_q_logic, jumped_lambda_logic = random.choice(potential_q_lambda_pairs)
        
    return int(jumped_q_logic), int(jumped_lambda_logic)


if __name__ == "__main__":
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    np.random.seed(5)
    fign = 1

    VEHICLE_MODEL = 'Dubin'
    OBSTACLE_MODE = 'dynamic'
    FUTURE_STEPS_DYNAMIC_OBSTACLE = 0
    USE_IMAGE=True
    NUMBER_OF_STATIC_OBSTACLES = 1
    OBSTACLE_MODE = 'static'
    USE_IMAGE = True
    IGNORE_OBSTACLES = False
    VEHICLE_MODEL = 'Dubin'

    noise_magnitude_position = 0.2+0*0.05
    noise_magnitude_orientation = 0*0.1
    vor_margin_factor = 0.75
    TIME_STEPS = 5000
    SAMPLING_TIME_SECONDS = 0.05
    collision_radius = 0.5
    capture_radius = 0.5

    normal_base_position = (-5, 0) 
    hyrlmp_base_position = ( 5, 0)

    STATIC_OBSTACLE_LOCATIONS_shared = [(0,0), (-2.75, 2.75), (2.75, 2.75), (2.75, -2.75), (-2.75, -2.75)]
    STATIC_OBSTACLE_LOCATIONS_normal = [hyrlmp_base_position] + STATIC_OBSTACLE_LOCATIONS_shared
    STATIC_OBSTACLE_LOCATIONS_hyrlmp = [normal_base_position] + STATIC_OBSTACLE_LOCATIONS_shared

    STATIC_OBSTACLE_RADII = [0.1] + [0.5]*len(STATIC_OBSTACLE_LOCATIONS_shared)

    normal_agent = PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_v4.zip")
    save_name = 'hyrl_sets/ob1_Mij_x00_y00_static_example11_v1.pkl' 
    with open(save_name, 'rb') as inp:
        HyRL_MP_sets = pickle.load(inp)
    hyrlmp_agent = [PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_jtrain0_v2.zip"),
                    PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_jtrain1_v2.zip")]
    
    HyRL_MP_sets.set_voronoi_partition(STATIC_OBSTACLE_LOCATIONS_hyrlmp, STATIC_OBSTACLE_RADII)
    HyRL_MP_sets.vor_margin_factor = vor_margin_factor
    HyRL_MP_sets.POSITIONAL_SET_POINT = normal_base_position
    NUMBER_OF_STATIC_OBSTACLES = len(STATIC_OBSTACLE_LOCATIONS_normal)
    
    kwargs= dict(initialize_state_randomly=False, 
                 TIME_STEPS=TIME_STEPS, 
                 train_hyrl=False, 
                 NUMBER_OF_STATIC_OBSTACLES=NUMBER_OF_STATIC_OBSTACLES,
                 clip_through_obstacles=False, 
                 SAMPLING_TIME_SECONDS=SAMPLING_TIME_SECONDS, 
                 train=False, 
                 RANDOM_NUMBER_OF_STATIC_OBSTACLES=False, 
                 STATIC_OBSTACLE_RADII=STATIC_OBSTACLE_RADII,
                 VEHICLE_MODEL=VEHICLE_MODEL,
                 IGNORE_OBSTACLES=IGNORE_OBSTACLES,
                 OBSTACLE_MODE=OBSTACLE_MODE,
                 USE_IMAGE=USE_IMAGE,
                 FUTURE_STEPS_DYNAMIC_OBSTACLE=FUTURE_STEPS_DYNAMIC_OBSTACLE,
                 setpoint_radius=capture_radius)

    theta_normal = np.arctan2(hyrlmp_base_position[1]-normal_base_position[1],hyrlmp_base_position[0]-normal_base_position[0])
    env_init_normal = np.array([normal_base_position[0],
                               normal_base_position[1],
                               np.cos(theta_normal),
                               np.sin(theta_normal)])
    theta_hyrlmp = np.arctan2(normal_base_position[1]-hyrlmp_base_position[1],normal_base_position[0]-hyrlmp_base_position[0])
    env_init_hyrlmp = np.array([hyrlmp_base_position[0],
                               hyrlmp_base_position[1],
                               np.cos(theta_hyrlmp),
                               np.sin(theta_hyrlmp)])
 
    ENV = BirdsEyeViewStaticObstacleLocationsEnvironment
    normal_env = ENV(INITIAL_STATE=env_init_normal,
                     STATIC_OBSTACLE_LOCATIONS=STATIC_OBSTACLE_LOCATIONS_normal,
                     POSITIONAL_SET_POINT=hyrlmp_base_position,
                      **kwargs)
    hyrlmp_env = ENV(INITIAL_STATE=env_init_hyrlmp,
                     STATIC_OBSTACLE_LOCATIONS=STATIC_OBSTACLE_LOCATIONS_hyrlmp,
                     POSITIONAL_SET_POINT=normal_base_position,
                    **kwargs)

    rule = "flow"
    t_max = TIME_STEPS*SAMPLING_TIME_SECONDS
    j_max = TIME_STEPS

    obs_normal, _ = normal_env.reset()
    obs_hyrlmp, _ = hyrlmp_env.reset()

    f = lambda x: flow( x, normal_env, hyrlmp_env)
    C = lambda x: inside_C( x, normal_env, hyrlmp_env)
    g = lambda x: jump( x, normal_env, hyrlmp_env, normal_agent, hyrlmp_agent, noise_magnitude_position, noise_magnitude_orientation, HyRL_MP_sets, normal_base_position, hyrlmp_base_position, collision_radius, capture_radius)
    D = lambda x: inside_D( x, normal_env, hyrlmp_env)

    initial_normal_action, _ = normal_agent.predict(obs_normal, deterministic=True)

    initial_hyrlmp_q_logic = random.choice([0, 1])
    initial_hyrlmp_lambda_logic = random.choice([*range(NUMBER_OF_STATIC_OBSTACLES)])
    initial_hyrlmp_vehicle_position = env_init_hyrlmp[0:2]
    initial_hyrlmp_q_logic, initial_hyrlmp_lambda_logic = jump_HyRL_MP_logic_variables(HyRL_MP_sets, initial_hyrlmp_vehicle_position, initial_hyrlmp_q_logic, initial_hyrlmp_lambda_logic)
    initial_hyrlmp_action, _ = hyrlmp_agent[initial_hyrlmp_q_logic].predict(obs_hyrlmp, deterministic=True)



    initial_condition = np.array([
        env_init_normal[0],
        env_init_normal[1],
        env_init_normal[2],
        env_init_normal[3],
        initial_normal_action[0],
        initial_normal_action[1],
        0,
        #
        env_init_hyrlmp[0],
        env_init_hyrlmp[1],
        env_init_hyrlmp[2],
        env_init_hyrlmp[3],
        initial_hyrlmp_action[0],
        initial_hyrlmp_action[1],
        initial_hyrlmp_q_logic, 
        initial_hyrlmp_lambda_logic,
        0,
        #
        0,
        1,
        0,
        0
        ])
    initial_condition = initial_condition.reshape(len(initial_condition), 1)
    xs_normal, ys_normal = [], []
    xs_hyrlmp, ys_hyrlmp = [], []
    ts = []
    final_point = TIME_STEPS

    capturetheflag_hybridsystem  = hybridsystem(f, C, g, D, initial_condition, rule, j_max, t_max, atol=1e-8, rtol=1e-8,)
    sol, ts, j = capturetheflag_hybridsystem.solve()     

    xs_normal = sol[0,:]
    ys_normal = sol[1,:]
    angle_normal = np.arctan2(sol[3,:], sol[2,:])
    has_enemy_flag_normal = sol[6,:]

    xs_hyrlmp = sol[7,:]
    ys_hyrlmp = sol[8,:]
    angle_hyrlmp = np.arctan2(sol[10,:], sol[9,:])
    has_enemy_flag_hyrlmp = sol[15,:]

    normal_points_scored = sol[18,:]
    hyrlmp_points_scored = sol[19,:]

    print('Normal agent scored ' + str(sol[18,-1]) + '!')
    print('HyRL-MP agent scored ' + str(sol[19,-1]) + '!')

    arrow = u'$\u2191$'
    markersize = 100
    timer_elapsed = 0
    number = 0
    for time_point in range(len(ts)):
        if time_point == 0:
            timer_elapsed += ts[time_point]
        else:
            timer_elapsed += ts[time_point]-ts[time_point-1]
        if timer_elapsed > SAMPLING_TIME_SECONDS*5: # set to SAMPLING_TIME_SECONDS*5 for making gifs
            timer_elapsed = 0 
            thetas_plot = np.linspace(0,2*np.pi, 100, endpoint=False)
            plt.plot(normal_base_position[0]+np.cos(thetas_plot)*capture_radius, normal_base_position[1]+np.sin(thetas_plot)*capture_radius, '--', color='red', linewidth=0.5)
            plt.plot(hyrlmp_base_position[0]+np.cos(thetas_plot)*capture_radius, hyrlmp_base_position[1]+np.sin(thetas_plot)*capture_radius, '--', color='blue', linewidth=0.5)

            normal_view = matplotlib.patches.Rectangle((xs_normal[time_point]-normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, ys_normal[time_point]-normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2), normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, 
            edgecolor='red', facecolor='none', rotation_point='center', angle=angle_normal[time_point]*180/np.pi, linestyle='--')
            hyrlmp_view = matplotlib.patches.Rectangle((xs_hyrlmp[time_point]-hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, ys_hyrlmp[time_point]-hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2), hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, 
            edgecolor='blue', facecolor='none', rotation_point='center', angle=angle_hyrlmp[time_point]*180/np.pi, linestyle='--')
            plt.gca().add_patch(normal_view)
            plt.gca().add_patch(hyrlmp_view)

            rotated_marker_normal = matplotlib.markers.MarkerStyle(marker=9)
            rotated_marker_normal._transform = rotated_marker_normal.get_transform().rotate_deg(angle_normal[time_point]*180/np.pi)
            rotated_marker_hyrlmp = matplotlib.markers.MarkerStyle(marker=9)
            rotated_marker_hyrlmp._transform = rotated_marker_hyrlmp.get_transform().rotate_deg(angle_hyrlmp[time_point]*180/np.pi)
            plt.scatter(xs_normal[time_point], ys_normal[time_point], marker=rotated_marker_normal, s=(markersize), facecolors='red')#,edgecolors='red')
            plt.scatter(xs_hyrlmp[time_point], ys_hyrlmp[time_point], marker=rotated_marker_hyrlmp, s=(markersize), facecolors='blue')#,edgecolors='blue')
            plt.scatter(xs_normal[time_point], ys_normal[time_point], marker='o', s=(markersize)/1, facecolors='red',edgecolors='red')
            plt.scatter(xs_hyrlmp[time_point], ys_hyrlmp[time_point], marker='o', s=(markersize)/1, facecolors='blue',edgecolors='blue')
            

            for idob, obstacle in enumerate(normal_env.STATIC_OBSTACLE_LOCATIONS[1:]):
                idob += 1
                x_obst, y_obst = obstacle[0], obstacle[1]
                obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                                radius=normal_env.STATIC_OBSTACLE_RADII[idob], color='gray')
                plt.gca().add_patch(obstaclePatch)

            if has_enemy_flag_normal[time_point] == False and has_enemy_flag_hyrlmp[time_point] == False:
                plt.plot([xs_normal[time_point], hyrlmp_base_position[0]], [ys_normal[time_point], hyrlmp_base_position[1]], '--', color='red', linewidth=0.5)
                plt.plot([xs_hyrlmp[time_point], normal_base_position[0]], [ys_hyrlmp[time_point], normal_base_position[1]], '--', color='blue', linewidth=0.5)
            elif has_enemy_flag_normal[time_point] == True and has_enemy_flag_hyrlmp[time_point] == False:
                plt.plot([xs_normal[time_point], normal_base_position[0]], [ys_normal[time_point], normal_base_position[1]], '--', color='red', linewidth=0.5)
                plt.plot([xs_hyrlmp[time_point], xs_normal[time_point]],   [ys_hyrlmp[time_point], ys_normal[time_point]],   '--', color='blue', linewidth=0.5)
            elif has_enemy_flag_normal[time_point] == False and has_enemy_flag_hyrlmp[time_point] == True:
                plt.plot([xs_hyrlmp[time_point], hyrlmp_base_position[0]], [ys_hyrlmp[time_point], hyrlmp_base_position[1]], '--', color='blue', linewidth=0.5)
                plt.plot([xs_hyrlmp[time_point], xs_normal[time_point]],   [ys_hyrlmp[time_point], ys_normal[time_point]],   '--', color='red', linewidth=0.5)
            else:
                plt.plot([xs_hyrlmp[time_point], xs_normal[time_point]],   [ys_hyrlmp[time_point], ys_normal[time_point]],   '--', color='black', linewidth=0.5)
            
            if has_enemy_flag_normal[time_point] == True:
                plt.plot(xs_normal[time_point]+0.1, ys_normal[time_point]+0.1, marker='$\Gamma$', markersize=10, color='blue', zorder=10)
            else:
                plt.plot(hyrlmp_base_position[0], hyrlmp_base_position[1], marker='$\Gamma$', markersize=10, color='blue', zorder=10)
                
            if has_enemy_flag_hyrlmp[time_point] == True:
                plt.plot(xs_hyrlmp[time_point]+0.1, ys_hyrlmp[time_point]+0.1, marker='$\Gamma$', markersize=10, color='red', zorder=10)
            else:
                plt.plot(normal_base_position[0], normal_base_position[1], marker='$\Gamma$', markersize=10, color='red', zorder=10)
            
            plt.text(-3, 4, 'Score '+str(round(normal_points_scored[time_point])), fontsize=15, fontweight='bold', color='red', horizontalalignment='center')
            plt.text( 3, 4, 'Score '+str(round(hyrlmp_points_scored[time_point])), fontsize=15, fontweight='bold', color='blue', horizontalalignment='center')

            plt.grid(visible=True)
            plt.xlabel('$p_x$', fontsize=22)
            plt.ylabel('$p_y$', fontsize=22)
            plt.xlim(-6, 6)
            plt.ylim([-6, 6])
            plt.tight_layout()

            
            plt.savefig('gifstorage/capturetheflag/capturetheflag_obstacles'+str(NUMBER_OF_STATIC_OBSTACLES)+'_noisemagposition'+str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+str(noise_magnitude_orientation).replace('.','')+
                        '_frame'+f'_n{number}'+'.png', 
                                transparent = False,  
                                facecolor = 'white'
                                )
            plt.pause(0.05)
            number += 1
            plt.clf()
    
    frames = []
    for idx in range(number):
        image = imageio.v2.imread('gifstorage/capturetheflag/capturetheflag_obstacles'+str(NUMBER_OF_STATIC_OBSTACLES)+'_noisemagposition'+str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+str(noise_magnitude_orientation).replace('.','')+'_frame'+f'_n{idx}'+'.png')
        frames.append(image)    
    imageio.mimsave('gifs/capturetheflag/capturetheflag_obstacles'+str(NUMBER_OF_STATIC_OBSTACLES)+'_noisemag'+'_noisemagposition'+str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+str(noise_magnitude_orientation).replace('.','')+'.gif', # output gif
                    frames,          # arr00ay00 of input frames
                    fps = 10)         # optional: frames per second    
