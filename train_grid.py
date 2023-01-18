import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import os
import statistics
from numpy.linalg import multi_dot
import gym
import gym_minigrid
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX 
from gym_minigrid.minigrid import Time_variant_only
from gym_minigrid.minigrid import lunarSurfaceTempThresholdUpper, lunarSurfaceTempThresholdLower, thermalThresholdUpper, thermalThresholdLower, batteryThresholdMin, batteryThresholdMax, slope_max_threshold
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, plot_results2, plot_results3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from gym_minigrid.simulation_param import xlength

randomPlayOn = False
trainingOn = False
testingOn = False
BesttestingOn = True

# max_env_steps must be smaller than max_env_steps_for_graphmaking defined in minigrid.py
env_name = 'MiniGrid-Empty-102x102-v0'
max_env_steps = 450
map_size = xlength 

# successful params
# training_eposodes = 40000
# learning_rate_set = 0.001  # default 0.0001
# learning_starts_set = 5000 # default 50000
# gamma_set = 0.99 # default 0.99
# exploration_fraction_set = 0.2 # default 0.1
# exploration_initial_eps_set = 1.0 # default 1.0
# exploration_final_eps_set = 0.05 # default 0.05
# log_frequency = 1000*10 # it was 2000 in earlier versions
# mean_average_num = 2*10-1
# rendering_step = 10

training_eposodes = 2000000
learning_rate_set = 0.0002  # default 0.0001
learning_starts_set = 100000 # default 50000
exploration_fraction_set = 0.1 # default 0.1
exploration_initial_eps_set = 1.0 # default 1.0
exploration_final_eps_set = 0.05 # default 0.05
gamma_set = 0.995 # default 0.99
tau_def = 0.10
net_arch_def=[64, 64, 64, 64]

log_frequency = 1000*5
mean_average_num = 2*5-1
rendering_step = 10

thermalThresholdUpper = 45 + 273.15 
thermalThresholdLower = 0 + 273.15
batteryThresholdMin = 60
batteryThresholdMax = 100

## Customize minigrid environment ##
# As you can see the environment's observation output also contains a rendered image as state representation (obs['image']]. However, for this exercise we want to train out deep RL agent from non-image state input (for faster convergence].
# We will create a slightly altered version of the FullyObsWrapper from the repository who's observation function returns a flat vector representing the full grid.

class FlatObsWrapper(gym.core.ObservationWrapper):
    """Fully observable gridworld returning a flat grid encoding."""

    def __init__(self, env):
        super().__init__(env)

        if Time_variant_only == True:
            self.observation_space = spaces.Dict({
              'position': gym.spaces.Box(low=0, high=map_size+10, shape=(2,), dtype='uint8'),
              # 'orientation': gym.spaces.Box(low=0, high=5, shape=(1,), dtype='uint8'),
              'timestep': gym.spaces.Box(low=0, high=max_env_steps+10, shape=(1,), dtype='int16'),
              # 'lunartemp': gym.spaces.Box(low=0, high=150, shape=(1,), dtype='float16'),
            })
        else:
            self.observation_space = spaces.Dict({
              'position': gym.spaces.Box(low=0, high=map_size+10, shape=(2,), dtype='uint8'),
              # 'orientation': gym.spaces.Box(low=0, high=5, shape=(1,), dtype='uint8'),
              'timestep': gym.spaces.Box(low=0, high=max_env_steps+10, shape=(1,), dtype='int16'),
              'thermal': gym.spaces.Box(low=-100, high=150, shape=(6,), dtype='float16'),
              'battery': gym.spaces.Box(low=0, high=100, shape=(1,), dtype='float16')
            })

        self.unwrapped.max_steps = max_env_steps

    def observation(self, obs):
        # this method is called in the step() function to get the observation
        # we provide code that gets the grid state and places the agent in it
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid = full_grid[1:-1, 1:-1]   # remove outer walls of the environment (for efficiency)
        flattened_grid = full_grid.flatten()
        # return flattened_grid

        panelLength = 6
        roverAllTempCurrent = np.zeros((panelLength))
        for i in range(panelLength):
            roverAllTempCurrent[i] = env.roverAllTempSelected[i,env.step_count]-273.15
        # print(roverAllTempCurrent)
        batPowerPercentageCurrent = env.BatteryPowerPercentage[env.step_count]
        lunarSurfaceTempCurrent = env.LStempHistory[env.step_count]-273.15

        if Time_variant_only == True:
            ob_array = {
            'position': np.array([np.uint8(env.agent_pos[0]), np.uint8(env.agent_pos[1])]),
            'timestep': np.array([np.int16(env.step_count)]),
            # 'lunartemp': np.array([np.float16(lunarSurfaceTempCurrent)]),
            }
        else:
            ob_array = {
            'position': np.array([np.uint8(env.agent_pos[0]), np.uint8(env.agent_pos[1])]),
            'timestep': np.array([np.int16(env.step_count)]),
            'thermal': np.array([np.float16(roverAllTempCurrent[0]),np.float16(roverAllTempCurrent[1]),np.float16(roverAllTempCurrent[2]),np.float16(roverAllTempCurrent[3]),np.float16(roverAllTempCurrent[4]),np.float16(roverAllTempCurrent[5])]),
            'battery': np.array([np.float16(batPowerPercentageCurrent)])
            }

        return ob_array

    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)

# Random agent - we only use it in this cell for demonstration
class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *unused_args):
        return self.action_space.sample(), None

# save training process
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``].

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

        self.filename = "monitorave.csv"
        f = open(self.filename, "w+")
        f.close()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')

          if len(x) > 10: # the first batch is sometimes erronous (I couldn't find the root cause of this phenomena), so start from the second batch

              # Mean training reward over the last mean_average_num episodes
              mean_reward = statistics.mean(y[-mean_average_num:]) 
              print('statistics:', mean_reward)

              with open(self.filename, 'a', newline='') as csvfile:
                 csvwriter = csv.writer(csvfile)
                 csvwriter.writerow([mean_reward])

              if self.verbose > 0:
                 print(f"Num timesteps: {self.num_timesteps}")
                 print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here

              if mean_reward > self.best_mean_reward:
                 self.best_mean_reward = mean_reward
                 # Example for saving best model
                 if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                 self.model.save(self.save_path)

        return True


## Create environment with flat observation ##
env = FlatObsWrapper(gym.make(env_name))
env_eval = FlatObsWrapper(gym.make(env_name))
check_env(env)

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = Monitor(env, log_dir)
env_eval = Monitor(env_eval, log_dir)


## Learning ##

if trainingOn:

    # Use the default hidden layer configuration 2 x 64
    # model = DQN('MultiInputPolicy', env, learning_rate=learning_rate_set, buffer_size=1000000, learning_starts=learning_starts_set, batch_size=64, tau=tau_def, gamma=gamma_set, exploration_fraction=exploration_fraction_set, exploration_initial_eps=exploration_initial_eps_set, exploration_final_eps=exploration_final_eps_set, verbose=1)

    # Custom policy of two layers of size 8 each with tanh activation function
    import tensorflow as tf
    policy_kwargs = dict(net_arch=net_arch_def) # my guess is replacing "net_arch" by "layers" should solve your issue) 
    model = DQN('MultiInputPolicy', env, learning_rate=learning_rate_set, buffer_size=1000000, learning_starts=learning_starts_set, batch_size=64, tau=tau_def, gamma=gamma_set, exploration_fraction=exploration_fraction_set, exploration_initial_eps=exploration_initial_eps_set, exploration_final_eps=exploration_final_eps_set, policy_kwargs=policy_kwargs, verbose=1)
    
    # evaluate untrained model
    mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # Create the callback: check every 'log_frequency' steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=log_frequency, log_dir=log_dir)

    model.learn(total_timesteps=int(training_eposodes), callback=callback, log_interval=10)

    # Save the agent
    model.save("dqn_grid")
    del model  # delete trained model to demonstrate loading

    # prepare mean values to plot
    CSVDataMeanReward = open("monitorave.csv")
    mean_reward_history = np.loadtxt(CSVDataMeanReward, delimiter=",")
    ypoints = np.arange(log_frequency, training_eposodes+log_frequency, log_frequency)

    plot_results([log_dir], training_eposodes, results_plotter.X_TIMESTEPS, "Grid map world")
    plt.plot(ypoints, mean_reward_history, "r-")
    plt.savefig('training_result.png')
    # plt.show()

    plot_results([log_dir], training_eposodes, results_plotter.X_TIMESTEPS, "Grid map world")
    plt.plot(ypoints, mean_reward_history, "r-")
    plt.ylim((-2000,1000))
    plt.savefig('training_result_closedLook.png')
    # plt.show()


## Evaluation ##

if testingOn:

    env = FlatObsWrapper(gym.make(env_name))
    env_eval = FlatObsWrapper(gym.make(env_name))

    model = DQN.load("dqn_grid", env=env)

    for repeat in range(1):

        # Enjoy trained agent
        observation = env.reset()

        done = False
        episode_reward = 0
        episode_length = 0
        for i in range(max_env_steps):
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if env.step_count % rendering_step == 0:
                env.render()
                print('action:', action)
                print('_states:', _states)
                print('observation:', observation)
                print('reward:', reward)

            if done == True:
                break

        print('Total reward:', episode_reward)
        print('Total length:', episode_length)

        episode_length = episode_length+1 # time shift for graph visualization

        plt.close()
        plt.clf()

        plt.close()
        env.roverTopTempSelected.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=thermalThresholdLower-273.15, y2=thermalThresholdUpper-273.15, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.roverTopTempSelected-273.15, s=10, c ="black")
        plt.xlabel('Time steps')
        plt.ylabel('Temperature (degC)')
        plt.title("Rover Top temperature history")
        plt.ylim((0,60))
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('last_top_temp.png')

        plt.close()
        env.BatteryPowerPercentage.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=batteryThresholdMin, y2=batteryThresholdMax, facecolor='lime', alpha=0.3)
        # plt.fill_between(x=x_axis, y1=batteryThresholdMin, y2=batteryThresholdMax, facecolor="#03AF7A", alpha=0.3)
        plt.scatter(x=x_axis, y=env.BatteryPowerPercentage, s=10, c ="black")
        plt.ylim((0,100))
        plt.xlim((1,episode_length-1))
        plt.title("Rover battery power history")
        plt.xlabel('Time steps')
        plt.ylabel('Remaining battery power (%)')
        plt.show()
        plt.savefig('last_bat_power.png')

        plt.close()
        env.rewardPos.resize((1, episode_length))
        env.slopepenalty.resize((1, episode_length))
        env.TPpenalty.resize((1, episode_length))
        env.thermalPenalty_.resize((1, episode_length))
        env.powerPenalty_.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.scatter(x=x_axis, y=env.rewardPos, s=10, c ="blue", label="pos")
        plt.scatter(x=x_axis, y=-env.slopepenalty, s=10, c ="green", label="slope")
        plt.scatter(x=x_axis, y=-env.thermalPenalty_, s=10, c ="red", label="thermal")
        plt.scatter(x=x_axis, y=-env.powerPenalty_, s=10, c ="orange", label="power")
        plt.scatter(x=x_axis, y=env.rewardPos-env.slopepenalty-env.TPpenalty, s=10, c ="black", label="total")
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('last_reward.png')

        print("episode_length",episode_length-1)
        print("env.rewardPos",env.rewardPos[0, episode_length-1])

        plt.close()
        env.rewardPos.resize((1, episode_length))
        env.slopepenalty.resize((1, episode_length))
        env.LStemppenalty.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.scatter(x=x_axis, y=env.rewardPos, s=10, c ="blue", label="pos")
        plt.scatter(x=x_axis, y=-env.slopepenalty, s=10, c ="green", label="slope")
        plt.scatter(x=x_axis, y=-env.LStemppenalty, s=10, c ="orange", label="surface")
        plt.scatter(x=x_axis, y=env.rewardPos-env.slopepenalty-env.LStemppenalty, s=10, c ="black", label="total")
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('last_reward_tv.png')

        plt.close()
        env.LStempHistory.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=lunarSurfaceTempThresholdLower-273.15, y2=lunarSurfaceTempThresholdUpper-273.15, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.LStempHistory-273.15, s=10, c ="black")
        plt.ylim((0,120))
        plt.xlim((1,episode_length-1))
        plt.title("Local lunar surface temperature history")
        plt.xlabel('Time steps')
        plt.ylabel('Local lunar surface temperature (degC)')
        plt.show()
        plt.savefig('last_lunar_surface.png')

        plt.close()
        env.SlopeHistory.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=0, y2=slope_max_threshold, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.SlopeHistory, s=10, c ="black")
        plt.ylim((0,20))
        plt.xlim((1,episode_length-1))
        plt.title("Local lunar surface slope history")
        plt.xlabel('Time steps')
        plt.ylabel('Local lunar surface slope (deg)')
        plt.show()
        plt.savefig('last_lunar_slope.png')

        plt.close()
        env.actionHistory.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.scatter(x=x_axis, y=env.actionHistory, s=10, c ="black")
        plt.xlabel('Time steps')
        plt.ylabel('Action (stay 0, up 1, right 2, down 3, left 4)')
        plt.title("Rover Action history")
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('last_action.png')

        if rendering_step == 1:
            plt.close()
            env.lowtemp.resize((1, episode_length))
            env.hightemp.resize((1, episode_length))
            env.avetemp.resize((1, episode_length))
            x_axis = np.arange(0, episode_length)
            plt.scatter(x=x_axis, y=env.lowtemp, s=10, c ="#005AFF", label="lowest")
            plt.scatter(x=x_axis, y=env.hightemp, s=10, c ='#FF4B00', label="highest")
            plt.scatter(x=x_axis, y=env.avetemp, s=10, c ="#03AF7A", label="average")
            plt.legend(loc="lower right")
            plt.xlabel('Time steps')
            plt.ylabel('Temperature (degC)')
            plt.ylim((-100,120))
            plt.xlim((1,350))
            plt.title("Lunar surface temperature transition")
            plt.show()
            plt.savefig('lunarsurface_temp.png')


        last_top_temp_graph = -np.ones(episode_length)
        last_bat_power_graph = -np.ones(episode_length)
        last_lunar_slope_graph = -np.ones(episode_length)

        # for step_count in range(0, episode_length):

        #     filenameA = 'last_top_temp%02d.jpg' % step_count
        #     filenameB = 'last_bat_power%02d.jpg' % step_count
        #     filenameC = 'last_lunar_slope%02d.jpg' % step_count

        #     last_top_temp_graph[step_count] = env.roverTopTempSelected[0][step_count]
        #     last_bat_power_graph[step_count] = env.BatteryPowerPercentage[0][step_count]
        #     last_lunar_slope_graph[step_count] = env.SlopeHistory[0][step_count]
            
        #     plt.close()
        #     # env.roverTopTempSelected.resize((1, step_count))
        #     x_axis = np.arange(0, episode_length)
        #     plt.fill_between(x=x_axis, y1=thermalThresholdLower-273.15, y2=thermalThresholdUpper-273.15, facecolor='lime', alpha=0.3)
        #     plt.scatter(x=x_axis, y=last_top_temp_graph-273.15, s=10, c ="black")
        #     plt.show()
        #     plt.xlabel('Time steps')
        #     plt.ylabel('Temperature (degC)')
        #     plt.title("Rover Top temperature history")
        #     plt.ylim((0,60))
        #     plt.xlim((1,episode_length))
        #     plt.savefig(filenameA)

        #     plt.close()
        #     # env.BatteryPowerPercentage.resize((1, step_count))
        #     x_axis = np.arange(0, episode_length)
        #     plt.fill_between(x=x_axis, y1=batteryThresholdMin, y2=batteryThresholdMax, facecolor='lime', alpha=0.3)
        #     plt.scatter(x=x_axis, y=last_bat_power_graph, s=10, c ="black")
        #     plt.ylim((0,100))
        #     plt.xlim((1,episode_length))
        #     plt.title("Rover battery power history")
        #     plt.xlabel('Time steps')
        #     plt.ylabel('Remaining battery power (%)')
        #     plt.show()
        #     plt.savefig(filenameB)

        #     plt.close()
        #     # env.SlopeHistory.resize((1, step_count))
        #     x_axis = np.arange(0, episode_length)
        #     plt.fill_between(x=x_axis, y1=0, y2=slope_max_threshold, facecolor='lime', alpha=0.3)
        #     plt.scatter(x=x_axis, y=last_lunar_slope_graph, s=10, c ="black")
        #     plt.ylim((0,20))
        #     plt.xlim((1,episode_length))
        #     plt.title("Local lunar surface slope history")
        #     plt.xlabel('Time steps')
        #     plt.ylabel('Local lunar surface slope (deg)')
        #     plt.show()
        #     plt.savefig(filenameC)

        if os.path.exists("savedimage_last.jpg"):
            os.remove("savedimage_last.jpg")

        # Absolute path of a file
        old_name = 'savedimage.jpg'
        new_name = 'savedimage_last.jpg'

        # Renaming the file
        os.rename(old_name, new_name)


if BesttestingOn:

    import shutil
    src_path = 'tmp/best_model.zip'
    dst_path = 'best_model.zip'
    shutil.copy(src_path, dst_path)
    print('Copied')

    env = FlatObsWrapper(gym.make(env_name))
    env_eval = FlatObsWrapper(gym.make(env_name))

    model = DQN.load("best_model", env=env)

    for repeat in range(1):

        # Enjoy trained agent
        observation = env.reset()

        done = False
        episode_reward = 0
        episode_length = 0
        for i in range(max_env_steps):
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if env.step_count % rendering_step == 0:
                env.render()
                print('action:', action)
                print('_states:', _states)
                print('observation:', observation)
                print('reward:', reward)

            if done == True:
                break

        print('Total reward:', episode_reward)
        print('Total length:', episode_length)

        plt.close()
        plt.clf()

        episode_length = episode_length+1 # time shift for graph visualization
        
        plt.close()
        env.roverTopTempSelected.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=thermalThresholdLower-273.15, y2=thermalThresholdUpper-273.15, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.roverTopTempSelected-273.15, s=10, c ="black")
        plt.show()
        plt.xlabel('Time steps')
        plt.ylabel('Temperature (degC)')
        plt.title("Rover Top temperature history")
        plt.ylim((0,60))
        plt.xlim((1,episode_length-1))
        plt.savefig('best_top_temp.png')

        plt.close()
        env.BatteryPowerPercentage.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=batteryThresholdMin, y2=batteryThresholdMax, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.BatteryPowerPercentage, s=10, c ="black")
        plt.ylim((0,100))
        plt.xlim((1,episode_length-1))
        plt.title("Rover battery power history")
        plt.xlabel('Time steps')
        plt.ylabel('Remaining battery power (%)')
        plt.show()
        plt.savefig('best_bat_power.png')

        plt.close()
        env.rewardPos.resize((1, episode_length))
        env.slopepenalty.resize((1, episode_length))
        env.TPpenalty.resize((1, episode_length))
        env.thermalPenalty_.resize((1, episode_length))
        env.powerPenalty_.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.scatter(x=x_axis, y=env.rewardPos, s=10, c ="blue", label="pos")
        plt.scatter(x=x_axis, y=-env.slopepenalty, s=10, c ="green", label="slope")
        plt.scatter(x=x_axis, y=-env.thermalPenalty_, s=10, c ="red", label="thermal")
        plt.scatter(x=x_axis, y=-env.powerPenalty_, s=10, c ="orange", label="power")
        plt.scatter(x=x_axis, y=env.rewardPos-env.slopepenalty-env.TPpenalty, s=10, c ="black", label="total")
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('best_reward.png')

        plt.close()
        env.rewardPos.resize((1, episode_length))
        env.slopepenalty.resize((1, episode_length))
        env.LStemppenalty.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.scatter(x=x_axis, y=env.rewardPos, s=10, c ="blue", label="pos")
        plt.scatter(x=x_axis, y=-env.slopepenalty, s=10, c ="green", label="slope")
        plt.scatter(x=x_axis, y=-env.LStemppenalty, s=10, c ="orange", label="surface")
        plt.scatter(x=x_axis, y=env.rewardPos-env.slopepenalty-env.LStemppenalty, s=10, c ="black", label="total")
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('best_reward_tv.png')

        plt.close()
        env.LStempHistory.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=lunarSurfaceTempThresholdLower-273.15, y2=lunarSurfaceTempThresholdUpper-273.15, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.LStempHistory-273.15, s=10, c ="black")
        plt.ylim((0,120))
        plt.xlim((1,episode_length-1))
        plt.title("Local lunar surface temp history")
        plt.xlabel('Time steps')
        plt.ylabel('Local lunar surface temp (degC)')
        plt.show()
        plt.savefig('best_lunar_surface.png')

        plt.close()
        env.SlopeHistory.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.fill_between(x=x_axis, y1=0, y2=slope_max_threshold, facecolor='lime', alpha=0.3)
        plt.scatter(x=x_axis, y=env.SlopeHistory, s=10, c ="black")
        plt.ylim((0,20))
        plt.xlim((1,episode_length-1))
        plt.title("Local lunar surface slope history")
        plt.xlabel('Time steps')
        plt.ylabel('Local lunar surface slope (deg)')
        plt.show()
        plt.savefig('best_lunar_slope.png')

        plt.close()
        env.actionHistory.resize((1, episode_length))
        x_axis = np.arange(0, episode_length)
        plt.scatter(x=x_axis, y=env.actionHistory, s=10, c ="black")
        plt.xlabel('Time steps')
        plt.ylabel('Action (stay 0, up 1, right 2, down 3, left 4)')
        plt.title("Rover Action history")
        plt.xlim((1,episode_length-1))
        plt.show()
        plt.savefig('best_action.png')

        if rendering_step == 1:
            plt.close()
            env.lowtemp.resize((1, episode_length))
            env.hightemp.resize((1, episode_length))
            env.avetemp.resize((1, episode_length))
            x_axis = np.arange(0, episode_length)
            plt.scatter(x=x_axis, y=env.lowtemp, s=10, c ="#005AFF", label="lowest")
            plt.scatter(x=x_axis, y=env.hightemp, s=10, c ='#FF4B00', label="highest")
            plt.scatter(x=x_axis, y=env.avetemp, s=10, c ="#03AF7A", label="average")
            plt.legend(loc="lower right")
            plt.xlabel('Time steps')
            plt.ylabel('Temperature (degC)')
            plt.ylim((-100,120))
            plt.xlim((1,350))
            plt.title("Lunar surface temperature transition")
            plt.show()
            plt.savefig('lunarsurface_temp.png')


        best_top_temp_graph = -np.ones(episode_length)
        best_bat_power_graph = -np.ones(episode_length)
        best_lunar_slope_graph = -np.ones(episode_length)

        # for step_count in range(0, episode_length):

        #     filenameA = 'best_top_temp%02d.jpg' % step_count
        #     filenameB = 'best_bat_power%02d.jpg' % step_count
        #     filenameC = 'best_lunar_slope%02d.jpg' % step_count

        #     best_top_temp_graph[step_count] = env.roverTopTempSelected[0][step_count]
        #     best_bat_power_graph[step_count] = env.BatteryPowerPercentage[0][step_count]
        #     best_lunar_slope_graph[step_count] = env.SlopeHistory[0][step_count]
            
        #     plt.close()
        #     # env.roverTopTempSelected.resize((1, step_count))
        #     x_axis = np.arange(0, episode_length)
        #     plt.fill_between(x=x_axis, y1=thermalThresholdLower-273.15, y2=thermalThresholdUpper-273.15, facecolor='lime', alpha=0.3)
        #     plt.scatter(x=x_axis, y=best_top_temp_graph-273.15, s=10, c ="black")
        #     plt.show()
        #     plt.xlabel('Time steps')
        #     plt.ylabel('Temperature (degC)')
        #     plt.title("Rover Top temperature history")
        #     plt.ylim((0,60))
        #     plt.xlim((1,episode_length))
        #     plt.savefig(filenameA)

        #     plt.close()
        #     # env.BatteryPowerPercentage.resize((1, step_count))
        #     x_axis = np.arange(0, episode_length)
        #     plt.fill_between(x=x_axis, y1=batteryThresholdMin, y2=batteryThresholdMax, facecolor='lime', alpha=0.3)
        #     plt.scatter(x=x_axis, y=best_bat_power_graph, s=10, c ="black")
        #     plt.ylim((0,100))
        #     plt.xlim((1,episode_length))
        #     plt.title("Rover battery power history")
        #     plt.xlabel('Time steps')
        #     plt.ylabel('Remaining battery power (%)')
        #     plt.show()
        #     plt.savefig(filenameB)

        #     plt.close()
        #     # env.SlopeHistory.resize((1, step_count))
        #     x_axis = np.arange(0, episode_length)
        #     plt.fill_between(x=x_axis, y1=0, y2=slope_max_threshold, facecolor='lime', alpha=0.3)
        #     plt.scatter(x=x_axis, y=best_lunar_slope_graph, s=10, c ="black")
        #     plt.ylim((0,20))
        #     plt.xlim((1,episode_length))
        #     plt.title("Local lunar surface slope history")
        #     plt.xlabel('Time steps')
        #     plt.ylabel('Local lunar surface slope (deg)')
        #     plt.show()
        #     plt.savefig(filenameC)

        if os.path.exists("savedimage_best.jpg"):
            os.remove("savedimage_best.jpg")

        # Absolute path of a file
        old_name = 'savedimage.jpg'
        new_name = 'savedimage_best.jpg'

        # Renaming the file
        os.rename(old_name, new_name)


## Test with a random player ##

if randomPlayOn:

    rand_policy = RandPolicy(FlatObsWrapper(gym.make(env_name)).action_space)
    observation = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
      # Take a step
        action = rand_policy.act(observation)[0]
        observation, reward, done, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        if env.step_count % rendering_step == 0:
            env.render()
            print('action:', action)
            print('observation:', observation)
            print('reward:', reward)

    print('Total reward:', episode_reward)
    print('Total length:', episode_length)

    plt.close()
    env.roverTopTempSelected.resize((1, episode_length))
    sizeOfMatrix = np.shape(env.roverTopTempSelected)
    x_axis = np.arange(0, episode_length)
    plt.fill_between(x=x_axis, y1=0, y2=45, facecolor='lime', alpha=0.3)
    plt.scatter(x=x_axis, y=env.roverTopTempSelected-273.15, c ="black")
    plt.show()
    plt.ylim((0,50))
    plt.savefig('top_temp.png')

    plt.close()
    env.BatteryPowerPercentage.resize((1, episode_length))
    sizeOfMatrix = np.shape(env.BatteryPowerPercentage)
    x_axis = np.arange(0, episode_length)
    plt.fill_between(x=x_axis, y1=70, y2=100, facecolor='lime', alpha=0.3)
    plt.scatter(x=x_axis, y=env.BatteryPowerPercentage, c ="black")
    plt.show()
    plt.savefig('bat_power.png')