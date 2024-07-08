from math import e
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import string
import csv
import asyncio
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from sumo_rl import SumoEnvironment
import statistics

import os
import optuna
from optuna.logging import set_verbosity, DEBUG
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from joblib import Parallel, delayed
from sumo_rl import SumoEnvironment  # Import your SumoEnvironment
import json
import re
import shutil

class Train:
    def __init__(self, 
                 net_file, 
                 saturation_proportions=[0.75, 0.75, 0.75, 0.75],
                 total_timesteps=10000, 
                 trained_model=None, 
                 num_of_episodes=10, 
                 n_eval_episodes=5,
                 seed=10, 
                 fix_seed=False, 
                 fix_ts=False,
                 learning_rate=0.0001, 
                 learning_starts=0, 
                 train_freq=1, 
                 target_update_interval=2000, 
                 exploration_initial_eps=0.05, 
                 exploration_final_eps=0.01, 
                 training_fraction=1, 
                 verbose=1, 
                 tripinfo=False, 
                 emissioninfo=False, 
                 buffer_size=200000, 
                 batch_size=256, 
                 gamma=0.99):
        
        self.tripinfo = tripinfo
        self.emissioninfo = emissioninfo
        self.net_file = net_file
        self.output_folder = "outputs"

        # 提取net_file
        net_file_basename = os.path.basename(net_file)
        net_numbers = re.findall(r'\d+', net_file_basename)
        # 构建net_info
        net_info_suffix = "_fixed" if fix_ts else ""
        self.net_info = "_".join(net_numbers) + net_info_suffix
        # 构建flow_info
        self.flow_info = "-".join([str(p) for p in saturation_proportions])
        # 构建output_folder
        self.output_folder = self.generate_result_folder()

        # 构建route_file
        self.prop = saturation_proportions
        self.capacity_straight = 2080
        self.capacity_left = 1411
        self.capacity_right = 1411
        self.green_time_proportion = (30 - 4) / 120
        self.convert_to_seconds = 1 / 3600
        self.volumes = self._generate_volumes()
        self.route_file = os.path.join(self.output_folder, "output.rou.xml")
        self.create_xml()

        self.csv_name = "dqn"
        self.tripinfo_name = "tripinfos.xml"
        self.out_csv_name = os.path.join(self.output_folder, self.csv_name)
        self.tripinfo_output_name = os.path.join(self.output_folder, self.tripinfo_name)
        self.tripinfo_cmd = f"--tripinfo {self.tripinfo_output_name}"
        self.total_timesteps = total_timesteps
        self.model_save_path = os.path.join(self.output_folder, "model.zip")
        self.num_of_episodes = num_of_episodes
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.fix_seed = fix_seed
        self.fix_ts = fix_ts
        
        # initialize the environment
        self.env = SumoEnvironment(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=self.out_csv_name,
            single_agent=True,
            use_gui=False,
            num_seconds=int(self.total_timesteps / self.num_of_episodes),
            sumo_seed=self.seed,
            fixed_seed=self.fix_seed,
            fixed_ts=self.fix_ts,
            tripinfo=self.tripinfo,
            emissioninfo=self.emissioninfo,
            output_folder=self.output_folder
        )
        self.trained_model = trained_model
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.training_fraction = training_fraction
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose

    def generate_result_folder(self):
        folfer_name = f"{self.net_info}_{self.flow_info}"
        base_dir = "autodl-tmp\\outputs"
        result_folder = os.path.join(base_dir, folfer_name)

        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)

        os.makedirs(result_folder)
        return result_folder


    def _generate_volumes(self):
        '''
        Generates the traffic volumes for each direction based on the proportion of saturations.

        Returns:
            list: The traffic volumes for each direction.
        '''
        n_propoertion, s_propoertion, w_propoertion, e_propoertion = self.prop
        volumes = [self.capacity_straight * n_propoertion, self.capacity_right * n_propoertion, self.capacity_left * n_propoertion,
                   self.capacity_straight * s_propoertion, self.capacity_left * s_propoertion, self.capacity_right * s_propoertion,
                   self.capacity_straight * w_propoertion, self.capacity_right * w_propoertion, self.capacity_left * w_propoertion,
                   self.capacity_straight * e_propoertion, self.capacity_left * e_propoertion, self.capacity_right * e_propoertion]
        volumes = [round(volume * self.green_time_proportion * self.convert_to_seconds, 3) for volume in volumes]
        return volumes
    
    def create_xml(self):
        '''
        This method creates an XML file with specified routes and flows for SUMO simulation.

        The method first defines the volumes for different traffic flows.
        Then, it creates the root element of the XML file.
        Next, it adds a vType element to specify the vehicle type.
        After that, it adds route elements for each route in the simulation.
        Finally, it adds flow elements with specified periods for each route.

        Args:
            None

        Returns:
            None
        '''
        ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en = self.volumes
        # 创建根元素
        root = ET.Element("routes")

        # 添加vType元素
        ET.SubElement(root, "vType", accel="2.6", decel="4.5", id="CarA", length="5.0", minGap="2.5", maxSpeed="55.55", sigma="0.5")

        # 添加route元素
        routes = ["n_t t_s", "n_t t_w", "n_t t_e", "s_t t_n", "s_t t_w", "s_t t_e", "w_t t_e", "w_t t_s", "w_t t_n", "e_t t_w", "e_t t_s", "e_t t_n"]
        for i, route in enumerate(routes, start=1):
            ET.SubElement(root, "route", id=f"route{i:02d}", edges=route)

        # 添加flow元素并替换period值
        periods = [ns, nw, ne, sn, sw, se, we, ws, wn, ew, es, en]
        for i, period in enumerate(periods, start=1):
            ET.SubElement(root, "flow", id=f"flow{i:02d}", begin="0", end="100000", period=f"exp({period})", route=f"route{i:02d}", type="CarA", color="1,1,0")

        # 创建树结构并进行格式化
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_string = reparsed.toprettyxml(indent="  ")

        # 将格式化后的XML写入文件
        with open(self.route_file, "w", encoding="utf-8") as f:
            f.write(pretty_string)

    def train(self):
        """
        Trains the model using the specified parameters.

        If a pre-trained model is provided, it loads the model and sets the environment.(It means continue training the model.)
        Otherwise, it creates a new DQN model with the specified parameters.(It means train a new model.)
        """
        if self.trained_model:
            model = DQN.load(self.trained_model)
            model.set_env(self.env)
        else:
            model = DQN(
                env=self.env,
                policy="MlpPolicy",
                learning_rate=self.learning_rate,
                learning_starts=self.learning_starts,
                train_freq=self.train_freq,
                target_update_interval=self.target_update_interval,
                exploration_initial_eps=self.exploration_initial_eps,
                exploration_final_eps=self.exploration_final_eps,
                exploration_fraction=self.training_fraction,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                verbose=self.verbose
            )

            model.learn(total_timesteps=self.total_timesteps)
            model.save(self.model_save_path)

    def evaluate(self):
        """
        Evaluates the trained model using the specified number of episodes.
        """
        if self.trained_model is None:
            raise ValueError("No trained model specified for evaluation.")
        
        model = DQN.load(self.trained_model)
        model.set_env(self.env)
        
        evaluate_policy(model, self.env, n_eval_episodes=self.n_eval_episodes)
    
    def optimize_hyperparameters(self, n_trials=5):
        # 设置 Optuna 的日志级别为 DEBUG 以输出更多过程信息
        set_verbosity(DEBUG)
        
        def objective(trial):
            
            env = SumoEnvironment(
                net_file=self.net_file,
                route_file=self.route_file,
                out_csv_name=self.out_csv_name,
                single_agent=True,
                use_gui=False,
                num_seconds=int(self.total_timesteps / self.num_of_episodes),
                sumo_seed=self.seed,
                fixed_seed=self.fix_seed,
                fixed_ts=self.fix_ts,
                tripinfo=self.tripinfo,
                emissioninfo=self.emissioninfo,
                output_folder=self.output_folder
            )
            env = DummyVecEnv([lambda: env])
            
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            learning_starts = trial.suggest_int('learning_starts', 0, 5000)
            train_freq = trial.suggest_int('train_freq', 1, 5)
            target_update_interval = trial.suggest_int('target_update_interval', 1000, 3000)
            exploration_initial_eps = trial.suggest_float('exploration_initial_eps', 0.04, 1.0)
            exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1)
            exploration_fraction = trial.suggest_float('exploration_fraction', 0.2, 0.8)
            buffer_size = trial.suggest_int('buffer_size', 50000, 500000)
            batch_size = trial.suggest_int('batch_size', 32, 256)
            
            model = DQN(
                env=env,
                policy="MlpPolicy",
                learning_rate=learning_rate,
                learning_starts=learning_starts,
                train_freq=train_freq,
                target_update_interval=target_update_interval,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                exploration_fraction=exploration_fraction,
                buffer_size=buffer_size,
                batch_size=batch_size,
                verbose=0
            )
            
            model.learn(total_timesteps=self.total_timesteps)

            # 在训练过程中记录反馈结果
            rewards = []
            obs = env.reset()
            step_count = 0
            done = False
            while not done and step_count < self.total_timesteps:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                step_count += 1

            print(f"Total steps taken: {step_count}")

            # 计算总奖励和平均奖励
            total_reward = sum(rewards)
            mean_reward = total_reward / len(rewards) if rewards else 0.0
            
            # Record the trial results
            trial_results = {
                'learning_rate': learning_rate,
                'learning_starts': learning_starts,
                'train_freq': train_freq,
                'target_update_interval': target_update_interval,
                'exploration_initial_eps': exploration_initial_eps,
                'exploration_final_eps': exploration_final_eps,
                'exploration_fraction': exploration_fraction,
                'buffer_size': buffer_size,
                'batch_size': batch_size,
            }
            
            return mean_reward
            

        # Create Optuna study and set serial optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best hyperparameters: {study.best_params}")

        # Update class attributes with the best hyperparameters found
        best_params = study.best_params
        self.learning_rate = best_params['learning_rate']
        self.learning_starts = best_params['learning_starts']
        self.train_freq = best_params['train_freq']
        self.target_update_interval = best_params['target_update_interval']
        self.exploration_initial_eps = best_params['exploration_initial_eps']
        self.exploration_final_eps = best_params['exploration_final_eps']
        self.training_fraction = best_params['exploration_fraction']
        self.buffer_size = best_params['buffer_size']
        self.batch_size = best_params['batch_size']

        # write those hyperparameters to a file, name contains the net file name
        with open(os.path.join(self.output_folder, f"hyperparameters_{os.path.basename(self.net_file)}.json"), "w") as f:
            json.dump(best_params, f, indent=4)

                

            # Create Optuna study and set serial optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            print(f"Best hyperparameters: {study.best_params}")

            # Update class attributes with the best hyperparameters found
            best_params = study.best_params
            self.learning_rate = best_params['learning_rate']
            self.learning_starts = best_params['learning_starts']
            self.train_freq = best_params['train_freq']
            self.target_update_interval = best_params['target_update_interval']
            self.exploration_initial_eps = best_params['exploration_initial_eps']
            self.exploration_final_eps = best_params['exploration_final_eps']
            self.training_fraction = best_params['exploration_fraction']
            self.buffer_size = best_params['buffer_size']
            self.batch_size = best_params['batch_size']

            # write those hyperparameters to a file, name contains the net file name
            with open(os.path.join(self.output_folder, f"hyperparameters_{os.path.basename(self.net_file)}.json"), "w") as f:
                json.dump(best_params, f, indent=4)



import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statistics

class ShowResults:
    def __init__(self, result_folder):
        self.result_folder = result_folder
        self.metrics = ['system_total_stopped', 'agents_total_accumulated_waiting_time', 'system_mean_waiting_time']
        self.csv_prefix = 'dqn_conn0'
        self.tripinfo_prefix = 'tripinfos'
        self.csv_file_paths = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.startswith(self.csv_prefix) and f.endswith('.csv')]
        self.tripinfo_file_paths = [os.path.join(result_folder, f) for f in os.listdir(result_folder) \
                                    if f.startswith(self.tripinfo_prefix) and f.endswith('.xml')]
        self.combined_df = self.concatenate_csv_files()
        self.combined_csv_path = os.path.join(self.result_folder, "combined.csv")
        self.combined_df.to_csv(self.combined_csv_path, index=False)
            
    def concatenate_csv_files(self):

        dfs = []
        for file_path in self.csv_file_paths:
            if os.path.exists(file_path):
                dfs.append(pd.read_csv(file_path))
            else:
                raise FileNotFoundError(f"File not found: {file_path}")

        # Add a cumulative step column to each dataframe
        for i in range(1, len(dfs)):
            dfs[i]['step'] += dfs[i-1]['step'].max()

        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def drawing_from_csv(self):
        # Load combined CSV file
        df = pd.read_csv(self.combined_csv_path)

        # Plot accumulated waiting time over time
        plt.figure()
        plt.plot(df['step'], df['agents_total_accumulated_waiting_time'], label='Accumulated Waiting Time')
        plt.xlabel('Time Step')
        plt.ylabel('Accumulated Waiting Time')
        plt.title('Accumulated Waiting Time over Time')
        plt.legend()
        plt.grid(True)
        accumulated_waiting_time_path = os.path.join(self.result_folder, 'accumulated_waiting_time.png')
        plt.savefig(accumulated_waiting_time_path)
        plt.close()

        # Plot total stopped vehicles over time
        plt.figure()
        plt.plot(df['step'], df['system_total_stopped'], label='Total Stopped Vehicles', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('Total Stopped Vehicles')
        plt.title('Total Stopped Vehicles over Time')
        plt.legend()
        plt.grid(True)
        total_stopped_path = os.path.join(self.result_folder, 'total_stopped_vehicles.png')
        plt.savefig(total_stopped_path)
        plt.close()

        # Plot mean waiting time over time
        plt.figure()
        plt.plot(df['step'], df['system_mean_waiting_time'], label='Mean Waiting Time', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Mean Waiting Time')
        plt.title('Mean Waiting Time over Time')
        plt.legend()
        plt.grid(True)
        mean_waiting_time_path = os.path.join(self.result_folder, 'mean_waiting_time.png')
        plt.savefig(mean_waiting_time_path)
        plt.close()

    def save_plot(self, steps, values, filename, ylabel, title):
        plt.figure()
        plt.plot(steps, values, label=ylabel)
        plt.xlabel('step')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    def calculate_statistics(self, values):
        average = statistics.mean(values)
        median = statistics.median(values)
        return average, median

    def main(self):
        # Log file setup
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("Execution started at {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Find the relevant CSV files in the result folder
            file_paths = sorted([os.path.join(self.result_folder, f) for f in os.listdir(self.result_folder) if f.startswith(self.file_prefix) and f.endswith('.csv')])

            # Ensure all file paths exist
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

            # Read and concatenate the CSV files, even if combined.csv exists
            combined_df = self.read_and_concatenate_csv(file_paths)
            combined_csv_path = os.path.join(self.result_folder, "combined.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            log_file.write("Combined CSV saved to {}\n".format(combined_csv_path))

            for metric in self.metrics:
                # Get steps and metric values from combined.csv
                steps = combined_df['step']
                values = combined_df[metric]
                
                combined_plot_filename = os.path.join(self.result_folder, f'{metric}_combined.png')
                self.save_plot(steps, values, combined_plot_filename, metric, f'{metric} (Combined)')
                log_file.write("Combined plot for {} saved to {}\n".format(metric, combined_plot_filename))

                # Calculate and log average and median of the selected metric
                avg_value, median_value = self.calculate_statistics(combined_df[metric])
                log_file.write(f'Overall average {metric}: {avg_value:.3f}\n')
                log_file.write(f'Overall median {metric}: {median_value:.3f}\n')
            
            # Calculate and log average value of the selected metric for each episode
            episode_averages = []
            for i, file_path in enumerate(file_paths, start=1):
                df = pd.read_csv(file_path)
                avg_value = df[self.metrics[1]].mean()
                episode_averages.append((i, avg_value))
                log_file.write(f'Average {self.metrics[1]} for episode {i}: {avg_value:.3f}\n')
    
# Example of usage
# processor = ShowResults(result_folder="path/to/results", log_file_path="path/to/log.txt")
# processor.main()


# 假设 TrafficMatrix, Train, 和 ShowResults 类已经定义好了

import subprocess

def upload_to_onedrive(local_file, remote_folder):
    '''
    Uploads a local file to a remote folder in OneDrive using rclone.
    Please note that the local file will be renamed with the parent folder name as a prefix before uploading.
    Args:
        local_file (str): The path of the local file to be uploaded.
        remote_folder (str): The path of the remote folder in OneDrive.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the rclone command fails to execute.

    '''
    # 获取母文件夹名称
    parent_folder_name = os.path.basename(os.path.dirname(local_file))
    
    # 获取目标文件名称
    file_name = os.path.basename(local_file)
    
    # 生成新的文件名称
    new_file_name = f"{parent_folder_name}-{file_name}"
    
    # 生成新的本地文件路径
    new_local_file_path = os.path.join(os.path.dirname(local_file), new_file_name)
    
    # 重命名文件
    os.rename(local_file, new_local_file_path)
    
    # 使用rclone上传文件
    rclone_command = f"rclone copyto {new_local_file_path} {remote_folder}/{new_file_name}"
    try:
        subprocess.run(rclone_command, check=True, shell=True)
        print(f"Successfully uploaded {new_local_file_path} to {remote_folder}/{new_file_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload {new_local_file_path} to {remote_folder}/{new_file_name}. Error: {e}")



