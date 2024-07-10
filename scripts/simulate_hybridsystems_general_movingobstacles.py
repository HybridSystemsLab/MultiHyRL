import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as clrs
import random
import imageio
from env_static_obstacles import BirdsEyeViewStaticObstacleLocationsEnvironment
from stable_baselines3 import PPO
from hybridequationsolver import  hybridsystem
from utils import Def_Mob_general
import pickle

def plot_Mi(position_vehicle, Mij, index_lambda, res=40, POSITIONAL_SET_POINT=(0,0), width_hyrlmaps=3):
    # j is either 0 or 1
    xp = np.linspace(-4,4, res)
    yp = np.linspace(-4, 4, res)
    xi,yi = np.meshgrid(xp,yp)
    in_Mi0 = np.zeros((res,res))
    in_Mi1 = np.zeros((res,res))
    for idx in range(res):
        for idy in range(res):
            point = (xp[idx], yp[idy])
            in_Mi0[idy, idx] = Mij.get_in_Mi(point, index_q=0, index_lambda=index_lambda) 
            in_Mi1[idy, idx] = Mij.get_in_Mi(point, index_q=1, index_lambda=index_lambda)
    in_Mi01 = in_Mi0*in_Mi1
    
    cmap = clrs.ListedColormap(['none', 'lightblue'])
    plt.scatter(xi, yi, s=15, c=in_Mi0, rasterized=True, cmap =cmap, zorder=-1)
    cmap = clrs.ListedColormap(['none', 'lightgreen'])
    plt.scatter(xi, yi, s=15, c=in_Mi1, rasterized=True, cmap =cmap, zorder=-1)
    cmap = clrs.ListedColormap(['none', 'lightcoral'])
    plt.scatter(xi, yi, s=15, c=in_Mi01, rasterized=True, cmap =cmap, zorder=-1)

def set_setpoints(time_elapsed):
    time_spaces = 10
    # sequence_setpoints = [(0, 0), ( 5, 0), 
    #                       (5, 5), ( 0,  5),
    #                       (-5, 5), (-5, 0),
    #                       (-5, -5), (0,  -5),
    #                       (5, -5), (-5, 0)]
    # sequence_setpoints = [( 3, 0), 
    #                       ( 0,  3),
    #                       (-3, 0),
    #                       (0,  -3),
    #                       (3, 0)]
    # sequence_setpoints = [(0, 0), ( 4, 4), 
    #                       (0, 2), ( -2,  -3),
    #                       (2, -3), ]
    # sequence_setpoints = [(-2, -4), (4, -4),
    #                       (2, 4), (-4, 4)]
    sequence_setpoints = [(3, .1), (-3, -.1), ]
                        #   (3, -1), (3, 1)]
    index_of_setpoints = int(np.floor(time_elapsed/time_spaces) % len(sequence_setpoints)) 
    HORIZONTAL_POSITIONAL_SET_POINT = sequence_setpoints[index_of_setpoints][0]
    VERTICAL_POSITIONAL_SET_POINT = sequence_setpoints[index_of_setpoints][1]
    return HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT

def dynamics_circling_obstacle(time_elapsed, initial_angle, radius_circling, period, circling_center):
    y_position = np.sin(2*np.pi/(period)*time_elapsed + initial_angle)*radius_circling + circling_center[1]
    x_position = 0*np.cos(2*np.pi/(period)*time_elapsed + initial_angle)*radius_circling + circling_center[0]
    # x_position = np.sin(2*np.pi/(period)*time_elapsed + initial_angle)*radius_circling + circling_center[0]
    return (x_position, y_position)

def initialize_environment(case_number, vor_margin_factor, ENV):
    np.random.seed(1)
    SAMPLING_TIME_SECONDS = 0.05
    TIME_STEPS = 200
    IGNORE_OBSTACLES = False
    VEHICLE_MODEL = 'Dubin'
    OBSTACLE_MODE = 'dynamic'
    POSITIONAL_SET_POINT = np.array([0.,-0])

    FUTURE_STEPS_DYNAMIC_OBSTACLE = 0
    USE_IMAGE=True

    NUMBER_OF_STATIC_OBSTACLES = 1
    OBSTACLE_MODE = 'static'
    USE_IMAGE = True
    IGNORE_OBSTACLES = False
    VEHICLE_MODEL = 'Dubin'
    
    save_name = 'hyrl_sets/ob1_Mij_x00_y00_static_example11_v1.pkl' 
    with open(save_name, 'rb') as inp:
        Mij = pickle.load(inp)
    Policies = [PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_jtrain0_v2.zip"),
                PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_jtrain1_v2.zip")]
    
    if case_number == 1:
        points_1, points_2 = 6, 5
        initial_angles = np.linspace(0,2*np.pi, points_1, endpoint=False).tolist() + \
            np.linspace(1/6*np.pi,(2+1/6)*np.pi, points_1, endpoint=False).tolist() + \
            np.linspace(0,2*np.pi, points_2, endpoint=False).tolist() + \
            np.linspace(0,2*np.pi, points_2, endpoint=False).tolist()
        radius_circlings = [9]*points_1 + [9]*points_1 + [3]*points_2 + [3]*points_2
        periods = [90]*points_1 + [-90]*points_1 + [30]*points_2 + [30]*points_2
        circling_centers = [(9,0)]*points_1 + [(-9, 0)]*points_1 + [(-6, 0)]*points_2 + [(6, 0)]*points_2
        STATIC_OBSTACLE_LOCATIONS = []
        for obstacle_index in range(len(initial_angles)):
            STATIC_OBSTACLE_LOCATIONS.append(dynamics_circling_obstacle(0, initial_angles[obstacle_index], 
                                                                    radius_circlings[obstacle_index], periods[obstacle_index], 
                                                                    circling_center=circling_centers[obstacle_index]))
        dynamic_obstacle_kwargs = (initial_angles, radius_circlings, periods, circling_centers)
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        alphas = [1*np.pi] # np.linspace(0, 2*np.pi, num=50, endpoint=False) # 60
        inits = []
        L = 6
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 2:
        points_1, points_2 = 3, 6
        initial_angles = np.linspace(0,2*np.pi, points_1, endpoint=False).tolist() + \
            np.linspace(1/6*np.pi,(2+1/6)*np.pi, points_2, endpoint=False).tolist() 
        radius_circlings = [2.]*points_1 + [5]*points_2 #+ [3]*points_2 + [3]*points_2
        periods = [30]*points_1 + [-60]*points_2 
        circling_centers = [(0,0)]*points_1 + [(0, 0)]*points_2
        STATIC_OBSTACLE_LOCATIONS = []
        for obstacle_index in range(len(initial_angles)):
            STATIC_OBSTACLE_LOCATIONS.append(dynamics_circling_obstacle(0, initial_angles[obstacle_index], 
                                                                    radius_circlings[obstacle_index], periods[obstacle_index], 
                                                                    circling_center=circling_centers[obstacle_index]))
        dynamic_obstacle_kwargs = (initial_angles, radius_circlings, periods, circling_centers)
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        alphas = [0.75*np.pi] # np.linspace(0, 2*np.pi, num=50, endpoint=False) # 60
        inits = []
        L = 6
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L/6*4
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 3:
        points_1, points_2 = 1, 1
        initial_angles = np.linspace(0,np.pi, points_1, endpoint=True).tolist() + \
            np.linspace(3/6*np.pi,(1+3/6)*np.pi, points_2, endpoint=True).tolist() 
        radius_circlings = [1]*points_1 + [1]*points_2 #+ [3]*points_2 + [3]*points_2
        periods = [12]*points_1 + [12]*points_2 
        circling_centers = [(-1.5,0.)]*points_1 + [(1.5, -0.)]*points_2
        STATIC_OBSTACLE_LOCATIONS = []
        for obstacle_index in range(len(initial_angles)):
            STATIC_OBSTACLE_LOCATIONS.append(dynamics_circling_obstacle(0, initial_angles[obstacle_index], 
                                                                    radius_circlings[obstacle_index], periods[obstacle_index], 
                                                                    circling_center=circling_centers[obstacle_index]))
        dynamic_obstacle_kwargs = (initial_angles, radius_circlings, periods, circling_centers)
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        alphas = [1*np.pi] # np.linspace(0, 2*np.pi, num=50, endpoint=False) # 60
        inits = []
        L = 4
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L/6*4
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    

    Normal_agent = PPO.load("agents/ppo_Dubin_static_futuresteps0_example11_v4.zip")
    
    Mij.set_voronoi_partition(STATIC_OBSTACLE_LOCATIONS, STATIC_OBSTACLE_RADII)
    Mij.vor_margin_factor = vor_margin_factor
    NUMBER_OF_STATIC_OBSTACLES = len(STATIC_OBSTACLE_LOCATIONS)
    NUMBER_OF_OBSTACLES = NUMBER_OF_STATIC_OBSTACLES
    
    kwargs = dict(initialize_state_randomly=False, 
                  TIME_STEPS=TIME_STEPS, 
                  train_hyrl=False, 
                  NUMBER_OF_STATIC_OBSTACLES=NUMBER_OF_STATIC_OBSTACLES,
                  POSITIONAL_SET_POINT=POSITIONAL_SET_POINT, 
                  clip_through_obstacles=False, 
                  SAMPLING_TIME_SECONDS=SAMPLING_TIME_SECONDS, 
                  train=False, 
                  RANDOM_NUMBER_OF_STATIC_OBSTACLES=False, 
                  STATIC_OBSTACLE_LOCATIONS=STATIC_OBSTACLE_LOCATIONS,
                  STATIC_OBSTACLE_RADII=STATIC_OBSTACLE_RADII,
                  VEHICLE_MODEL=VEHICLE_MODEL,
                  IGNORE_OBSTACLES=IGNORE_OBSTACLES,
                  OBSTACLE_MODE=OBSTACLE_MODE,
                  USE_IMAGE=USE_IMAGE,
                  FUTURE_STEPS_DYNAMIC_OBSTACLE=FUTURE_STEPS_DYNAMIC_OBSTACLE,)
    env = ENV(**kwargs)
    return env, Normal_agent, Policies, Mij, L, inits, NUMBER_OF_OBSTACLES, kwargs, min_x, min_y, max_x, max_y, dynamic_obstacle_kwargs


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

def flow( x, env, HyRL_MP=False):
    horizontal_position_vehicle    = x[0]
    vertical_position_vehicle     = x[1]
    orientation_vector_vehicle     = x[2:4]
    control_input_forward_velocity = x[4]
    control_input_orientation      = x[5]
    timer                          = x[6]
    noise_sign                     = x[7]
    time_elapsed                   = x[8]
    if HyRL_MP == True:
        q_logic                    = x[9]
        lambda_logic               = x[10]
    

    xdot = np.zeros( len(x) )
    xdot[0] =  control_input_forward_velocity*env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*orientation_vector_vehicle[0]
    xdot[1] =  control_input_forward_velocity*env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*orientation_vector_vehicle[1]
    xdot[2] =  control_input_orientation*env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*orientation_vector_vehicle[1]
    xdot[3] = -control_input_orientation*env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*orientation_vector_vehicle[0]
    xdot[4] =  0
    xdot[5] =  0
    xdot[6] =  1
    xdot[7] =  0
    xdot[8] =  1
    if HyRL_MP == True:
        xdot[9] = 0
        xdot[10] = 0
    return xdot

def jump( x, env, agent, noise_magnitude_position, noise_magnitude_orientation, HyRL_MP=False, HyRL_MP_sets=None, scale_observation=False,
         initial_angles=None, radius_circlings=None, periods=None, circling_centers=None):
    horizontal_position_vehicle    = x[0]
    vertical_position_vehicle     = x[1]
    orientation_vector_vehicle     = x[2:4]
    control_input_forward_velocity = x[4]
    control_input_orientation      = x[5]
    timer                          = x[6]
    noise_sign                     = x[7]
    time_elapsed                   = x[8]
    noise = np.array([noise_sign*noise_magnitude_position, -noise_sign*noise_magnitude_position, noise_sign*noise_magnitude_orientation])
    new_obstacle_locations = []
    for obstacle_index in range(len(initial_angles)):
        new_obstacle_locations.append(dynamics_circling_obstacle(time_elapsed, initial_angles[obstacle_index], 
                                                                 radius_circlings[obstacle_index], periods[obstacle_index],
                                                                 circling_center=circling_centers[obstacle_index]))
    env.STATIC_OBSTACLE_LOCATIONS = new_obstacle_locations
    HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT = set_setpoints(time_elapsed)
    env.HORIZONTAL_POSITIONAL_SET_POINT, env.VERTICAL_POSITIONAL_SET_POINT = HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT
    
    if HyRL_MP == True:
        q_logic                    = int(x[9])
        lambda_logic               = int(x[10])
        HyRL_MP_sets.POSITIONAL_SET_POINT = np.array([HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT])
        HyRL_MP_sets.set_voronoi_partition(new_obstacle_locations)
        perturbed_vehicle_position = (horizontal_position_vehicle+noise[0], vertical_position_vehicle+noise[1])
        jumped_q_logic, jumped_lambda_logic = jump_HyRL_MP_logic_variables(HyRL_MP_sets, perturbed_vehicle_position, q_logic, lambda_logic)
        agent = agent[jumped_q_logic]
    
    noisy_observation = perturb_observation(env, noise, horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    noisy_observation['vector'][0] = noisy_observation['vector'][0]
    action, _ = agent.predict(noisy_observation, deterministic=True)

    jumped_horizontal_position_vehicle    = horizontal_position_vehicle
    jumped_vertical_position_vehicle     = vertical_position_vehicle
    jumped_orientation_vector_vehicle     = orientation_vector_vehicle
    jumped_control_input_forward_velocity = action[0]
    jumped_control_input_orientation      = action[1]
    jumped_timer                          = 0
    jumped_noise_sign                     = noise_sign*(-1)

    jumped_x = [jumped_horizontal_position_vehicle,
                jumped_vertical_position_vehicle,
                jumped_orientation_vector_vehicle[0],
                jumped_orientation_vector_vehicle[1],
                jumped_control_input_forward_velocity,
                jumped_control_input_orientation,
                jumped_timer,
                jumped_noise_sign,
                time_elapsed]
    if HyRL_MP == True:
        jumped_x = jumped_x + [jumped_q_logic, jumped_lambda_logic]
    return jumped_x

def inside_C(x, env, HyRL_MP=False):
    horizontal_position_vehicle    = x[0]
    vertical_position_vehicle      = x[1]
    orientation_vector_vehicle     = x[2:4]
    control_input_forward_velocity = x[4]*env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY
    control_input_orientation      = x[5]*env.MAGNITUDE_CONTROL_INPUT_ORIENTATION
    timer                          = x[6]
    noise_sign                     = x[7]
    time_elapsed                   = x[8]
    if HyRL_MP == True:
        q_logic                    = x[9]
        lambda_logic               = x[10]

    # Check if bounds of the environment are hit:
    env.check_bounds(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    # Check if set point is reached:
    observation = env.update_observation(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    reached_set_point = env.check_set_point(observation)

    if (timer <= env.SAMPLING_TIME_SECONDS):# and (env.out_of_bounds == False):# and (reached_set_point == False):
        inside = 1
    else:
        inside = 0
    return inside

def inside_D(x, env, HyRL_MP=False):
    horizontal_position_vehicle    = x[0]
    vertical_position_vehicle      = x[1]
    orientation_vector_vehicle     = x[2:4]
    control_input_forward_velocity = x[4]*env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY
    control_input_orientation      = x[5]*env.MAGNITUDE_CONTROL_INPUT_ORIENTATION
    timer                          = x[6]
    noise_sign                     = x[7]
    time_elapsed                   = x[8]
    if HyRL_MP == True:
        q_logic                    = x[9]
        lambda_logic               = x[10]

    # Check if bounds of the environment are hit:
    env.check_bounds(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    # Check if set point is reached:
    observation = env.update_observation(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    reached_set_point = env.check_set_point(observation)

    if (timer >= env.SAMPLING_TIME_SECONDS):# and (env.out_of_bounds == False):# and (reached_set_point == False):
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
    
    j_inits = [0, 1]
    noise_combinations = [(0, 0), (0.2, 0)]
    sample_and_hold = True
    fign = 1
    for noise_combination in noise_combinations:
        noise_magnitude_position, noise_magnitude_orientation = noise_combination[0], noise_combination[1]
        print('noise case: ', noise_magnitude_position, noise_magnitude_orientation)
        for case_number in [3]:
            print('case number', case_number)
            vor_margin_factor = 0.75

            ENV = BirdsEyeViewStaticObstacleLocationsEnvironment
            env, Normal_agent, HyRL_Agent, HyRL_MP_sets, L, inits, NUMBER_OF_OBSTACLES, env_kwargs, min_x, min_y, max_x, max_y, dynamic_obstacle_kwargs = initialize_environment(case_number, vor_margin_factor, ENV)
            (initial_angles, radius_circlings, periods, circling_centers) = dynamic_obstacle_kwargs
            
            rule = "flow"
            t_max = 1*env_kwargs['TIME_STEPS']*env_kwargs['SAMPLING_TIME_SECONDS'] #5
            print('Maximum time ', t_max)
            j_max = 200*env_kwargs['TIME_STEPS']


            colors = ['blue', 'red']
            linestyles = ['--', '-']
            simcolors = ['blue', 'magenta']
            
            
            for init in inits:
                normal_env = ENV(INITIAL_STATE=init, **env_kwargs)
                hyrlmp_env = ENV(INITIAL_STATE=init, **env_kwargs)
                obs_normal, _ = normal_env.reset()
                obs_hyrlmp, _ = hyrlmp_env.reset()


                f_normal = lambda x: flow( x, normal_env)
                C_normal = lambda x: inside_C( x, normal_env)
                g_normal = lambda x: jump( x, normal_env, Normal_agent, noise_magnitude_position, noise_magnitude_orientation, initial_angles=initial_angles, radius_circlings=radius_circlings, periods=periods, circling_centers=circling_centers)
                D_normal = lambda x: inside_D( x, normal_env )
                initial_action_normal, _ = Normal_agent.predict(obs_normal, deterministic=True)
                additional_initial_conditions_normal = np.concatenate((initial_action_normal, np.array([0, -1, 0])))

                initial_condition_normal = np.concatenate((init, additional_initial_conditions_normal))
                initial_condition_normal = initial_condition_normal.reshape(len(initial_condition_normal), 1)

                f_hyrl_mp = lambda x: flow( x, hyrlmp_env, HyRL_MP=True)
                C_hyrl_mp = lambda x: inside_C( x, hyrlmp_env, HyRL_MP=True)
                g_hyrl_mp = lambda x: jump( x, hyrlmp_env, HyRL_Agent, noise_magnitude_position, noise_magnitude_orientation, HyRL_MP=True, HyRL_MP_sets=HyRL_MP_sets, initial_angles=initial_angles, radius_circlings=radius_circlings, periods=periods, circling_centers=circling_centers)
                D_hyrl_mp = lambda x: inside_D( x, hyrlmp_env, HyRL_MP=True )
                initial_q_logic = random.choice([0, 1])
                initial_lambda_logic = random.choice([*range(NUMBER_OF_OBSTACLES)])
                initial_vehicle_position = init[0:2]
                initial_q_logic, initial_lambda_logic = jump_HyRL_MP_logic_variables(HyRL_MP_sets, initial_vehicle_position, initial_q_logic, initial_lambda_logic)
                initial_action_hyrlmp, _ = HyRL_Agent[initial_q_logic].predict(obs_hyrlmp, deterministic=True)
                additional_initial_conditions_hyrlmp = np.concatenate((initial_action_hyrlmp, np.array([0, -1, 0, initial_q_logic, initial_lambda_logic])))

                initial_condition_hyrlmp = np.concatenate((init, additional_initial_conditions_hyrlmp))
                initial_condition_hyrlmp = initial_condition_hyrlmp.reshape(len(initial_condition_hyrlmp), 1)

                
                hybridsystem_normal  = hybridsystem(f_normal, C_normal, g_normal, D_normal, initial_condition_normal, rule, j_max, t_max, atol=1e-8, rtol=1e-7,)
                sol_normal, ts_normal, j = hybridsystem_normal.solve()  
                
                hybridsystem_hyrl_mp  = hybridsystem(f_hyrl_mp, C_hyrl_mp, g_hyrl_mp, D_hyrl_mp, initial_condition_hyrlmp, rule, j_max, t_max, atol=1e-7, rtol=1e-7,)
                sol_hyrlmp, ts_hyrlmp, j = hybridsystem_hyrl_mp.solve()        

                xs_normal = sol_normal[0,:]
                ys_normal = sol_normal[1,:]
                angle_normal = np.arctan2(sol_normal[3,:], sol_normal[2,:])

                xs_hyrlmp = sol_hyrlmp[0,:]
                ys_hyrlmp = sol_hyrlmp[1,:]
                angle_hyrlmp = np.arctan2(sol_hyrlmp[3,:], sol_hyrlmp[2,:])
                indexes_lambda = sol_hyrlmp[10,:]

                markersize = 100
                timer_elapsed = 0
                number = 0
                max_time = max(ts_normal[-1], ts_hyrlmp[-1])
                time_elapsed = 0
                while time_elapsed < max_time:
                    time_point_normal = np.absolute(ts_normal-time_elapsed).argmin()
                    
                    time_point_hyrlmp = np.absolute(ts_hyrlmp-time_elapsed).argmin()
                    
                    normal_view = matplotlib.patches.Rectangle((xs_normal[time_point_normal]-normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, ys_normal[time_point_normal]-normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2), normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, normal_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, 
                    edgecolor='red', facecolor='none', rotation_point='center', angle=angle_normal[time_point_normal]*180/np.pi, linestyle='--')
                    hyrlmp_view = matplotlib.patches.Rectangle((xs_hyrlmp[time_point_hyrlmp]-hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2, ys_hyrlmp[time_point_hyrlmp]-hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE/2), hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, hyrlmp_env.WIDTH_AND_LENGTH_BIRDSEYEVIEW_IMAGE, 
                    edgecolor='blue', facecolor='none', rotation_point='center', angle=angle_hyrlmp[time_point_hyrlmp]*180/np.pi, linestyle='--')
                    plt.gca().add_patch(normal_view)
                    plt.gca().add_patch(hyrlmp_view)

                    rotated_marker_normal = matplotlib.markers.MarkerStyle(marker=9)
                    rotated_marker_normal._transform = rotated_marker_normal.get_transform().rotate_deg(angle_normal[time_point_normal]*180/np.pi)
                    rotated_marker_hyrlmp = matplotlib.markers.MarkerStyle(marker=9)
                    rotated_marker_hyrlmp._transform = rotated_marker_hyrlmp.get_transform().rotate_deg(angle_hyrlmp[time_point_hyrlmp]*180/np.pi)
                    plt.scatter(xs_normal[time_point_normal], ys_normal[time_point_normal], marker=rotated_marker_normal, s=(markersize), facecolors='red')#,edgecolors='red')
                    plt.scatter(xs_hyrlmp[time_point_hyrlmp], ys_hyrlmp[time_point_hyrlmp], marker=rotated_marker_hyrlmp, s=(markersize), facecolors='blue')#,edgecolors='blue')
                    plt.scatter(xs_normal[time_point_normal], ys_normal[time_point_normal], marker='o', s=(markersize)/1, facecolors='red',edgecolors='red')
                    plt.scatter(xs_hyrlmp[time_point_hyrlmp], ys_hyrlmp[time_point_hyrlmp], marker='o', s=(markersize)/1, facecolors='blue',edgecolors='blue')
                    
                    obstacle_locations = []
                    for idob in range(NUMBER_OF_OBSTACLES):
                        
                        (x_obst, y_obst) = dynamics_circling_obstacle(ts_normal[time_point_normal], initial_angles[idob], radius_circlings[idob], periods[idob],
                                                                      circling_centers[idob])
                        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                                        radius=normal_env.STATIC_OBSTACLE_RADII[idob], color='gray', zorder=10)
                        plt.gca().add_patch(obstaclePatch)
                        obstacle_locations.append((x_obst, y_obst))
                    
                    HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT = set_setpoints(time_elapsed)
                    HyRL_MP_sets.POSITIONAL_SET_POINT = np.array([HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT])
                    HyRL_MP_sets.set_voronoi_partition(obstacle_locations)
                    plot_Mi((xs_hyrlmp[time_point_hyrlmp], ys_hyrlmp[time_point_hyrlmp]), HyRL_MP_sets, int(indexes_lambda[time_point_hyrlmp]),res=30) # 300l
                    plt.plot(HORIZONTAL_POSITIONAL_SET_POINT, VERTICAL_POSITIONAL_SET_POINT, '*', color='red', zorder=15, markersize=12)
                    plt.text(HORIZONTAL_POSITIONAL_SET_POINT+.1, VERTICAL_POSITIONAL_SET_POINT+.4, r'$p^*$', fontsize=22, color='red', zorder=15)
                    
                    
                    plt.grid(visible=True)
                    plt.xlabel('$p_x$', fontsize=22)
                    plt.ylabel('$p_y$', fontsize=22)
                    plt.xlim(-4, 4)
                    plt.ylim([-4, 4])
                    plt.tight_layout()

                    
                    plt.savefig('gifstorage/dynamic/dynamic_obstacles_casenumber'+str(case_number)+'_noisemagposition'+str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+str(noise_magnitude_orientation).replace('.','')+'_frame'+f'_n{number}'+
                                # '.pdf', 
                                '.png', 
                                        transparent = False,  
                                        facecolor = 'white'
                                        )
                    plt.pause(0.05)
                    number += 1
                    plt.clf()


                    time_elapsed += env_kwargs['SAMPLING_TIME_SECONDS']*5

                frames = []
                for idx in range(number):
                    image = imageio.v2.imread('gifstorage/dynamic/dynamic_obstacles_casenumber'+str(case_number)+'_noisemagposition'+str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+str(noise_magnitude_orientation).replace('.','')+'_frame'+f'_n{idx}'+'.png')
                    frames.append(image)    
                imageio.mimsave('gifs/dynamic/dynamic_obstacles_casenumber'+str(case_number)+'_noisemag'+'_noisemagposition'+str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+str(noise_magnitude_orientation).replace('.','')+'.gif', # output gif
                                frames,          # arr00ay00 of input frames
                                fps = 10)         # optional: frames per second    


