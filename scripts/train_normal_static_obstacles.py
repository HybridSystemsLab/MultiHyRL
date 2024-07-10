import torch as th
import gymnasium as gym
from torch import nn
import matplotlib.pyplot as plt
import pickle


from env_static_obstacles import BirdsEyeViewStaticObstacleLocationsEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results

from utils import find_critical_pointsV4, Def_Mob_general, Vor_part


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We assume CxHxW images (channels first)
                # Re-ordering will be done by pre-preprocessing or wrapper
                # new
                n_input_channels = subspace.shape[0]
                cnn = nn.Sequential(
                    nn.Conv2d(in_channels=n_input_channels, out_channels=2,
                              kernel_size=6, stride=2, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Flatten(),
                    )
                
                extractors[key] = cnn
                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                        ).shape[1]                 
                    # print(n_flatten)
                total_concat_size += n_flatten
            elif key == "vector":
                # Run through a simple MLP
                out_features_MLP = 32
                linear = nn.Sequential(nn.Linear(
                    in_features=subspace.shape[0], 
                    out_features=out_features_MLP), 
                    nn.ReLU(),
                    )
                extractors[key] = linear
                
                total_concat_size += out_features_MLP

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.        
        return th.cat(encoded_tensor_list, dim=1)
    
cwd = '' 
log_dir = cwd+"tmp/"
print('directory = ', log_dir)
# Create and wrap the environment

# Training parameters
TIME_STEPS = 200 
RANDOM_NUMBER_OF_STATIC_OBSTACLES=False
train = True
save = True
load = False
TOTAL_TRAINING_TIME_STEPS= 5000000 #
ENTROPY_COEFFICIENT=0.001
NUMBER_OF_PARALLEL_ENVS = 8

# Hybrid:
train_hyrl = False
Mij_complete = None
INITIAL_FOCUSED_OBSTACLE = None
init_in_whole_set_hyrl = True # init near boundary if false
margin = 0.1*2
# j_train = 1

# Obstacle settings
NUMBER_OF_STATIC_OBSTACLES = 4
IGNORE_OBSTACLES = False
VEHICLE_MODEL = 'Dubin'
OBSTACLE_MODE = 'dynamic'
ex_number = 11

STATIC_OBSTACLE_LOCATIONS = []
STATIC_OBSTACLE_RADII = 0.5
DYNAMIC_OBSTACLE_LOCATIONS=[],
NUMBER_OF_DYNAMIC_OBSTACLES=0
DYNAMIC_OBSTACLE_RADII=0.5
FUTURE_STEPS_DYNAMIC_OBSTACLE = 0
USE_IMAGE=True


NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS = []
NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS = []
HORIZONTAL_MOVEMENT_AMPLITUDES = []
VERTICAL_MOVEMENT_AMPLITUDES = []
HORIZONTAL_MOVEMENT_PERIODS = []
VERTICAL_MOVEMENT_PERIODS = []
DYNAMIC_OBSTACLE_MOTION = 'deterministic'
USE_TIMER=False
EVOLVE_TIME=False

for j_train in [1]:
    # general policy idea with observation scaling
    NUMBER_OF_STATIC_OBSTACLES = 1
    STATIC_OBSTACLE_LOCATIONS = [(-2, 0)]
    STATIC_OBSTACLE_RADII = 0.5
    OBSTACLE_MODE = 'static'
    USE_IMAGE = True
    IGNORE_OBSTACLES = False
    VEHICLE_MODEL = 'Dubin'
    # ENTROPY_COEFFICIENT=0
    load_name = cwd+'hyrl_sets/ob1_Mij_x00_y00_static_example11_v1.pkl' 
    with open(load_name, 'rb') as inp:
        Mij_complete = (pickle.load(inp))  


    env_kwargs = dict(TIME_STEPS=TIME_STEPS, 
                        RANDOM_NUMBER_OF_STATIC_OBSTACLES=RANDOM_NUMBER_OF_STATIC_OBSTACLES, 
                        NUMBER_OF_STATIC_OBSTACLES=NUMBER_OF_STATIC_OBSTACLES,
                        clip_through_obstacles=True,
                        IGNORE_OBSTACLES=IGNORE_OBSTACLES,
                        VEHICLE_MODEL=VEHICLE_MODEL, 
                        STATIC_OBSTACLE_LOCATIONS=STATIC_OBSTACLE_LOCATIONS, 
                        STATIC_OBSTACLE_RADII=STATIC_OBSTACLE_RADII,
                        train_hyrl=train_hyrl,
                        Mij_complete=Mij_complete,
                        INITIAL_FOCUSED_OBSTACLE = INITIAL_FOCUSED_OBSTACLE,
                        j_train=j_train,
                        init_in=init_in_whole_set_hyrl,
                        margin=margin,
                        DYNAMIC_OBSTACLE_LOCATIONS=DYNAMIC_OBSTACLE_LOCATIONS,
                        NUMBER_OF_DYNAMIC_OBSTACLES=NUMBER_OF_DYNAMIC_OBSTACLES,
                        DYNAMIC_OBSTACLE_RADII=DYNAMIC_OBSTACLE_RADII,
                        OBSTACLE_MODE=OBSTACLE_MODE,
                        USE_IMAGE=USE_IMAGE,
                        FUTURE_STEPS_DYNAMIC_OBSTACLE=FUTURE_STEPS_DYNAMIC_OBSTACLE,
                        DYNAMIC_OBSTACLE_MOTION=DYNAMIC_OBSTACLE_MOTION,
                        NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS=NUMBER_OF_HORIZONTAL_TERMS_OBSTACLE_DYNAMICS,
                        NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS=NUMBER_OF_VERTICAL_TERMS_OBSTACLE_DYNAMICS,
                        HORIZONTAL_MOVEMENT_AMPLITUDES=HORIZONTAL_MOVEMENT_AMPLITUDES,
                        VERTICAL_MOVEMENT_AMPLITUDES=VERTICAL_MOVEMENT_AMPLITUDES,
                        HORIZONTAL_MOVEMENT_PERIODS=HORIZONTAL_MOVEMENT_PERIODS,
                        VERTICAL_MOVEMENT_PERIODS=VERTICAL_MOVEMENT_PERIODS,
                        USE_TIMER=USE_TIMER,
                        EVOLVE_TIME=EVOLVE_TIME)

    env = make_vec_env(BirdsEyeViewStaticObstacleLocationsEnvironment,
                    n_envs=NUMBER_OF_PARALLEL_ENVS,
                    monitor_dir=log_dir,
                    env_kwargs=env_kwargs)

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[32, 32], vf=[32, 32]))
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=1e-4, gamma=0.99, ent_coef=ENTROPY_COEFFICIENT) 

    if load:
        print("Loading model....")
        LOAD_NAME_AGENT = "ppo_"+VEHICLE_MODEL+"_"+OBSTACLE_MODE+"_futuresteps"+str(FUTURE_STEPS_DYNAMIC_OBSTACLE)+"_example"+str(ex_number)+"_v1"
        if train_hyrl:
            LOAD_NAME_AGENT = LOAD_NAME_AGENT.replace("example"+str(ex_number), "example"+str(ex_number)+"_jtrain"+str(j_train)) 
        loaded_model = PPO.load(cwd+"agents/"+LOAD_NAME_AGENT)
        loaded_params = loaded_model.get_parameters()
        model.set_parameters(loaded_params)

    if train:
        # Separate evaluation env
        eval_env = BirdsEyeViewStaticObstacleLocationsEnvironment(**env_kwargs)
        # Stop training when the model reaches the reward threshold
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-0, verbose=1)
        eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
        print("Training model....")
        model.learn(total_timesteps=TOTAL_TRAINING_TIME_STEPS, callback=eval_callback)
        if save:
            print("Saving model....")
            save_name = "ppo_"+VEHICLE_MODEL+"_"+OBSTACLE_MODE+"_futuresteps"+str(FUTURE_STEPS_DYNAMIC_OBSTACLE)+"_example"+str(ex_number)+"_v2"
            if train_hyrl:
                save_name = save_name.replace("example"+str(ex_number), "example"+str(ex_number)+"_jtrain"+str(j_train)) 
            model.save(cwd+"agents/"+save_name)

    plot_results([log_dir], TOTAL_TRAINING_TIME_STEPS, results_plotter.X_TIMESTEPS, "PPO Bird's-eye view "+OBSTACLE_MODE+" obstacles")
    plt.ylim([-100, 0])
    plt.savefig(cwd+'plots/'+save_name)
