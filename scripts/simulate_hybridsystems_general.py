import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from env_static_obstacles import BirdsEyeViewStaticObstacleLocationsEnvironment
from stable_baselines3 import PPO
from hybridequationsolver import  hybridsystem
from utils import Def_Mob_general
import pickle



def initialize_environment(case_number, vor_margin_factor, ENV, random_radii=False):
    np.random.seed(1)
    SAMPLING_TIME_SECONDS = 0.05
    TIME_STEPS = 200
    IGNORE_OBSTACLES = False
    VEHICLE_MODEL = 'Dubin'
    OBSTACLE_MODE = 'dynamic'
    POSITIONAL_SET_POINT = np.array([0.,0])

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
        L = 4
        max_x, min_x =  0, -L-.5
        max_y, min_y =  1.2, -1.2 #4, -4
        STATIC_OBSTACLE_LOCATIONS = [(-2, 0)]
        STATIC_OBSTACLE_RADII = [0.5]
        alphas = np.linspace(1.05*np.pi, 0.95*np.pi, num=7, endpoint=True) 
        for alpha in alphas:
            init_cond = np.array([-1, np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
            init_cond[2], init_cond[3] = 1, 0 # np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    
    elif case_number == 2:
        # 2nd example
        STATIC_OBSTACLE_LOCATIONS = [(-1.7, 2.6),
                                    (-2.4, -1.9),
                                    (-4, 1.4)]
        alpha_bounds = [[0.6*np.pi, 0.85*np.pi], [1.05*np.pi, 1.45*np.pi],  [0.75*np.pi, 1.05*np.pi]]
        max_x, min_x = -1, -6
        max_y, min_y =  4.5, -4 #4, -4
        inits = []
        for idx, STATIC_OBSTACLE_LOCATION in enumerate(STATIC_OBSTACLE_LOCATIONS):
            alphas = np.linspace(alpha_bounds[idx][0], alpha_bounds[idx][1], 20, endpoint=True)
            for alpha in alphas:
                init_cond = np.array([min(max(np.cos(alpha)*L + STATIC_OBSTACLE_LOCATION[0] , min_x), max_x), 
                                    min(max(np.sin(alpha)*L + STATIC_OBSTACLE_LOCATION[1], min_y), max_y), 
                                    0, 0], dtype=np.float32)       
                theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
                init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
                inits.append(init_cond)

    elif case_number == 3:
        STATIC_OBSTACLE_LOCATIONS = [(-1.2, -1.2),
                                    (-2.2, 2)]
        STATIC_OBSTACLE_RADII = [0.4, 0.8]
        max_x, min_x = 0.5, -4.1
        max_y, min_y =  4, -3.1
        inits = []
        alpha_bounds = [[1.15*np.pi, 1.35*np.pi], [0.6*np.pi, 0.9*np.pi]]
        for idx, STATIC_OBSTACLE_LOCATION in enumerate(STATIC_OBSTACLE_LOCATIONS):
            alphas = np.linspace(alpha_bounds[idx][0], alpha_bounds[idx][1], 20, endpoint=True) #20
            for alpha in alphas:
                init_cond = np.array([min(max(np.cos(alpha)*L + STATIC_OBSTACLE_LOCATION[0] , min_x), max_x), 
                                min(max(np.sin(alpha)*L + STATIC_OBSTACLE_LOCATION[1], min_y), max_y), 
                                0, 0], dtype=np.float32)       
                theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
                init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
                inits.append(init_cond)

    elif case_number == 4:
        # 9 obstacle example hard
        STATIC_OBSTACLE_LOCATIONS = [(0.35057379297857, 2.188375927859161),
                                (-1.7134133806946483, 1.4575856669467404),
                                (-1.1589092842677644, -1.5661304581425155),
                                (1.7864247018307355, -1.0748871045596133),
                                (4.8139447220373865, 1.3512720722224645),
                                (0.9778348766807966, -4.903451738719028),
                                (-4.146928971098447, -2.7933814828383854),
                                (-4.107682516774461, 2.850779602386939),
                                (1.6723758862824596, 4.71202280289272)]
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        alphas = np.linspace(0*np.pi, 2*np.pi, num=200, endpoint=False) 
        inits = []
        L = 7
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 5:
        # 12 obstacles hard
        STATIC_OBSTACLE_LOCATIONS = [(1.0659098652892163, 1.76211725699931),
                                (-1.8418554867405013, 1.1279573887169625),
                                (-1.0578568182206252, -1.6664832419982727),
                                (1.7393365533607767, -0.8153839157303949),
                                (4.928382526031488, 0.8432352442275497),
                                (2.9800716318454814, -4.014868997746977),
                                (-1.521342452289482, -4.762931570247659),
                                (-3.9783682690481204, -3.028627728168496),
                                (-4.9774352586849595, -0.47448735030534683),
                                (-3.8570063202900293, 3.18174515718069),
                                (0.6260947951187125, 4.960645654299977),
                                (3.443030189194816, 3.625678297407688)]
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        alphas = np.linspace(-1/16*np.pi, -1/4*np.pi, num=200, endpoint=False)
        inits = []
        L = 7
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 6:
        STATIC_OBSTACLE_LOCATIONS = [(2.1, 2.1), (2.1, -2.1), (-2.1, -2.1), (-2.1, 2.1),
                                    (4.5, 0), (0, -4.5), (-4.5, 0), (0, 4.5),
                                    ]
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        if random_radii == True:
            STATIC_OBSTACLE_LOCATIONS = [(2.3, 2.3), (2.3, -2.3), (-2.3, -2.3), (-2.3, 2.3),
                                    (5., 0), (0, -5.), (-5., 0), (0, 5.),
                                    ]
            # STATIC_OBSTACLE_RADII = list(np.random.uniform(0.4, 0.8, len(STATIC_OBSTACLE_LOCATIONS)))
            STATIC_OBSTACLE_RADII = [0.5, 0.6, 0.4, 0.4, 
                                     0.75, 0.6, 0.9, 0.8]
        alphas = np.linspace(0*np.pi, 2*np.pi, num=100, endpoint=False) 
        inits = []
        L = 7
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 7:
        STATIC_OBSTACLE_LOCATIONS = [(0, 2), (2, -2.1), (-2, -2.1),
                                    (4.2, 2), (2.1, 4.7), (-2.1, 4.7), (-4.2, 2), (0, -5),
                                    (5, -2.3), (-5, -2.3),
                            ]
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        if random_radii == True:
            STATIC_OBSTACLE_RADII = list(np.random.uniform(0.4, 0.8, len(STATIC_OBSTACLE_LOCATIONS)))
        alphas = np.linspace(0*np.pi, 2*np.pi, num=100, endpoint=False)
        inits = []
        L = 7
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 8:
        STATIC_OBSTACLE_LOCATIONS = [(-2., -2.), (-2., 2.), (2., 2.), (2., -2.)]
                            
        STATIC_OBSTACLE_RADII = [0.5]*len(STATIC_OBSTACLE_LOCATIONS)
        # alphas = np.linspace(0*np.pi, 2*np.pi, num=100, endpoint=False) #60 [np.pi*0.36] # np.linspace(0, 2*np.pi, num=50, endpoint=False) # 60
        alphas = np.array([0.25*np.pi, 0.75*np.pi, 1.25*np.pi, 1.75*np.pi])
        alphas = np.sort(np.concatenate((alphas-0.02*np.pi, alphas, alphas+0.02*np.pi)))
        inits = []
        L = 5
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*(L+1)
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
            init_cond[2], init_cond[3] = np.cos(theta), np.sin(theta)
            inits.append(init_cond)
    elif case_number == 9:
        STATIC_OBSTACLE_LOCATIONS = [(0, 2.5), (3.5, -2.), (-3.5, -2.),
                    (4.5, 4.5), (0, -5), (-4.5, 4.5),
                                    ]
        STATIC_OBSTACLE_RADII = [0.5, 0.9, 0.8, 0.7, 0.4, 0.7]
        alphas = np.linspace(0*np.pi, 2*np.pi, num=100, endpoint=False) 
        L = 8
        max_x, min_x = L, -L
        max_y, min_y = L, -L
        inits = []
        for alpha in alphas:
            init_cond = np.array([np.cos(alpha), np.sin(alpha), 0, 0], dtype=np.float32)*L
            theta = np.arctan2(POSITIONAL_SET_POINT[1]-init_cond[1],POSITIONAL_SET_POINT[0]-init_cond[0])
            # theta += 0.95*np.pi
            # theta += np.random.uniform(low=-np.pi/1, high=np.pi/1)
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
    return env, Normal_agent, Policies, Mij, L, inits, NUMBER_OF_OBSTACLES, kwargs, min_x, min_y, max_x, max_y

def perturb_observation(env, noise, horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle):
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
    if HyRL_MP == True:
        q_logic                    = x[8]
        lambda_logic               = x[9]

    xdot = np.zeros( len(x) )
    xdot[0] =  control_input_forward_velocity*env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*orientation_vector_vehicle[0]
    xdot[1] =  control_input_forward_velocity*env.MAGNITUDE_CONTROL_INPUT_FORWARD_VELOCITY*orientation_vector_vehicle[1]
    xdot[2] =  control_input_orientation*env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*orientation_vector_vehicle[1]
    xdot[3] = -control_input_orientation*env.MAGNITUDE_CONTROL_INPUT_ORIENTATION*orientation_vector_vehicle[0]
    xdot[4] =  0
    xdot[5] =  0
    xdot[6] =  1
    xdot[7] =  0
    if HyRL_MP == True:
        xdot[8] = 0
        xdot[9] = 0
    return xdot

def jump( x, env, agent, noise_magnitude_position, noise_magnitude_orientation, HyRL_MP=False, HyRL_MP_sets=None, scale_observation=False):
    horizontal_position_vehicle    = x[0]
    vertical_position_vehicle      = x[1]
    orientation_vector_vehicle     = x[2:4]
    control_input_forward_velocity = x[4]
    control_input_orientation      = x[5]
    timer                          = x[6]
    noise_sign                     = x[7]
    noise = np.array([noise_sign*noise_magnitude_position, -noise_sign*noise_magnitude_position, noise_sign*noise_magnitude_orientation])

    if HyRL_MP == True:
        q_logic                    = int(x[8])
        lambda_logic               = int(x[9])
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
                jumped_noise_sign]
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
    if HyRL_MP == True:
        q_logic                    = x[8]
        lambda_logic               = x[9]

    # Check if bounds of the environment are hit:
    env.check_bounds(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    # Check if set point is reached:
    observation = env.update_observation(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    reached_set_point = env.check_set_point(observation)

    if (timer <= env.SAMPLING_TIME_SECONDS) and (env.out_of_bounds == False) and (reached_set_point == False):
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
    if HyRL_MP == True:
        q_logic                    = x[8]
        lambda_logic               = x[9]

    # Check if bounds of the environment are hit:
    env.check_bounds(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    # Check if set point is reached:
    observation = env.update_observation(horizontal_position_vehicle, vertical_position_vehicle, orientation_vector_vehicle)
    reached_set_point = env.check_set_point(observation)

    if (timer >= env.SAMPLING_TIME_SECONDS) and (env.out_of_bounds == False) and (reached_set_point == False):
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
    noise_combinations = [(0, 0), (0.2, 0), (0, 0.3), (0.05, 0.1)]
    sample_and_hold = True
    fign = 1
    for noise_combination in noise_combinations:
        noise_magnitude_position, noise_magnitude_orientation = noise_combination[0], noise_combination[1]
        print('noise case: ', noise_magnitude_position, noise_magnitude_orientation)
        for case_number in [9]:
            print('case number', case_number)
            vor_margin_factor = 0.75
            if case_number == 9:
                random_radii = True #True
            else:
                random_radii = False
            ENV = BirdsEyeViewStaticObstacleLocationsEnvironment
            np.random.seed(1)
            env, Normal_agent, HyRL_Agent, HyRL_MP_sets, L, inits, NUMBER_OF_OBSTACLES, env_kwargs, min_x, min_y, max_x, max_y = initialize_environment(case_number, vor_margin_factor, ENV, random_radii)
            
            
            rule = "flow"
            t_max = env_kwargs['TIME_STEPS']*env_kwargs['SAMPLING_TIME_SECONDS']
            j_max = env_kwargs['TIME_STEPS']


            colors = ['blue', 'red']
            linestyles = ['--', '-']
            if case_number == 1:
                simcolors = ['blue', 'blue', 'blue', 'blue', 'red', 'green', 'green', 'green', 'green']
            else:
                simcolors = ['blue', 'magenta']
           
            for cc, agent in enumerate([HyRL_Agent,  Normal_agent]):
                counter = 1
                
                SC = 0
                plt.figure(fign)
                fign += 1
                print(' Agent number ', cc)
                for init in inits:
                    print(counter)
                    counter += 1

                    env = ENV(INITIAL_STATE=init, **env_kwargs)
                    obs, _ = env.reset()
                    

                    if cc == 0:
                        f_hyrl_mp = lambda x: flow( x, env, HyRL_MP=True)
                        C_hyrl_mp = lambda x: inside_C( x, env, HyRL_MP=True)
                        g_hyrl_mp = lambda x: jump( x, env, agent, noise_magnitude_position, noise_magnitude_orientation, HyRL_MP=True, HyRL_MP_sets=HyRL_MP_sets)
                        D_hyrl_mp = lambda x: inside_D( x, env, HyRL_MP=True )
                        initial_q_logic = random.choice([0, 1])
                        initial_lambda_logic = random.choice([*range(NUMBER_OF_OBSTACLES)])
                        initial_vehicle_position = init[0:2]
                        initial_q_logic, initial_lambda_logic = jump_HyRL_MP_logic_variables(HyRL_MP_sets, initial_vehicle_position, initial_q_logic, initial_lambda_logic)
                        initial_action, _ = agent[initial_q_logic].predict(obs, deterministic=True)
                        additional_initial_conditions = np.concatenate((initial_action, np.array([0, -1, initial_q_logic, initial_lambda_logic])))
                    if cc == 1:
                        f_normal = lambda x: flow( x, env)
                        C_normal = lambda x: inside_C( x, env)
                        g_normal = lambda x: jump( x, env, agent, noise_magnitude_position, noise_magnitude_orientation )
                        D_normal = lambda x: inside_D( x, env )
                        initial_action, _ = agent.predict(obs, deterministic=True)
                        additional_initial_conditions = np.concatenate((initial_action, np.array([0, -1])))

                    initial_condition = np.concatenate((init, additional_initial_conditions))
                    initial_condition = initial_condition.reshape(len(initial_condition), 1)
                    xs, ys = [], []
                    ts = []
                    final_point = env_kwargs['TIME_STEPS']
                    reward_total = 0
                    if cc == 0:
                        hybridsystem_hyrl_mp  = hybridsystem(f_hyrl_mp, C_hyrl_mp, g_hyrl_mp, D_hyrl_mp, initial_condition, rule, j_max, t_max, atol=1e-7, rtol=1e-7,)
                        sol, ts, j = hybridsystem_hyrl_mp.solve()     
                    else:
                        hybridsystem_normal  = hybridsystem(f_normal, C_normal, g_normal, D_normal, initial_condition, rule, j_max, t_max, atol=1e-8, rtol=1e-7,)
                        sol, ts, j = hybridsystem_normal.solve()     

                    xs = sol[0,:]
                    ys = sol[1,:]


                    if env_kwargs['OBSTACLE_MODE'] == 'static':
                        plt.plot(xs,ys, linestyle=linestyles[0], color=simcolors[SC])
                        plt.plot(xs[-1], ys[-1], 'o', color=simcolors[SC], markersize=16,
                                fillstyle='none')
                        plt.plot(xs[0], ys[0], 'x', color=simcolors[SC], markersize=10)
                        # plt.plot(jumpsx, jumpsy, '*', color='red', markersize=16)

                    if case_number == 1:
                        if SC > len(simcolors):
                            SC = 0
                        else:
                            SC += 1
                    else:
                        SC = abs(SC-1)
                
                if env_kwargs['OBSTACLE_MODE'] == 'static':
                    plt.plot(env.HORIZONTAL_POSITIONAL_SET_POINT, env.VERTICAL_POSITIONAL_SET_POINT, '*', color='red', zorder=15, markersize=12)
                    plt.text(env.HORIZONTAL_POSITIONAL_SET_POINT+0.1,env.VERTICAL_POSITIONAL_SET_POINT+.4, r'$p^*$', fontsize=22, color='red', zorder=15)
                    for idob, obstacle in enumerate(env.STATIC_OBSTACLE_LOCATIONS):
                        x_obst, y_obst = obstacle[0], obstacle[1]
                        obstaclePatch = matplotlib.patches.Circle((x_obst,y_obst), 
                                                        radius=env.STATIC_OBSTACLE_RADII[idob], color='gray')
                        plt.gca().add_patch(obstaclePatch)
                    plt.grid(visible=True)
                    plt.xlabel('$p_x$', fontsize=22)
                    plt.ylabel('$p_y$', fontsize=22)
                    plt.xlim(min_x-0.3, max_x+0.3)
                    plt.ylim([min_y-0.3, max_y+0.3])
                    plt.tight_layout()
                

                if cc == 0:
                    save_add = 'hyrl'
                else:
                    save_add = 'normal'
                if random_radii == True:
                    save_add = 'randomradii_'+save_add
                plt.savefig('plots/applied_hyrl_sampleandhold_general_casenumber'+str(case_number)+
                            '_vormaring'+str(vor_margin_factor).replace('.','')+'_noisemagposition'+
                            str(noise_magnitude_position).replace('.','')+'_noisemagorienation'+
                            str(noise_magnitude_orientation).replace('.','')+'dubin_agent'+save_add+'.pdf')

