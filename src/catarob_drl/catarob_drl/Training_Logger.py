# DataLogger to keep track of training and test information for comparisons 
# focussing on performance metrics for data analysis, debugging and monitoring

import os
import time
import numpy as np


class DataLogger:
    def __init__(self, base_path, algorithm, stage, is_training=True):
        self.base_path = base_path                                                                              # root directory for logs
        self.algorithm = algorithm
        self.stage = stage                                                                                      # TODO (current stage of training/testing)
        self.is_training = is_training                                                                          # whether its training or testing session
        self.log_dir = os.path.join(self.base_path, 'logs', self.algorithm)
        self.ensure_directory_exists(self.log_dir)
        self.current_time = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = self.init_log_file()                                                                    # timestamp + main log file. creates lists to store episode data

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def init_log_file(self):
        mode = "train" if self.is_training else "test"
        filename = f"{self.algorithm}_{mode}_stage{self.stage}_{self.current_time}.csv"                         # header row of csv file
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            f.write("Episode,Reward,Length,Success,AvgReward,SuccessRate\n")
        return filepath

    def log_episode(self, episode, reward, length, success):                                                    #TODO
        self.episode_rewards.append(reward)                                                                     # model to log data for each episode  
        self.episode_lengths.append(length)                                                                     # reward and success rate over last 100 episodes
        self.episode_successes.append(success)

        avg_reward = np.mean(self.episode_rewards[-100:])
        success_rate = np.mean(self.episode_successes[-100:])

        with open(self.log_file, 'a') as f:
            f.write(f"{episode},{reward},{length},{success},{avg_reward:.2f},{success_rate:.2f}\n")

        if episode % 10 == 0:                                                                                   # every 10 episodes prints log summary to console
            print(f"Episode: {episode}, Reward: {reward:.2f}, Length: {length}, "
                  f"Success: {success}, Avg Reward: {avg_reward:.2f}, "
                  f"Success Rate: {success_rate:.2f}")

    def log_training_info(self, episode, actor_loss, critic_loss, epsilon):                                     #TODO
        training_log_file = os.path.join(self.log_dir, f"{self.algorithm}_training_info_stage{self.stage}_{self.current_time}.csv")          #seperate csv file for training
        if not os.path.exists(training_log_file):
            with open(training_log_file, 'w') as f:
                f.write("Episode,ActorLoss,CriticLoss,Epsilon\n")
        
        with open(training_log_file, 'a') as f:
            actor_loss_str = f"{actor_loss:.6f}" if actor_loss is not None else "None"
            f.write(f"{episode},{actor_loss_str},{critic_loss:.6f},{epsilon:.4f}\n")

    def log_test_results(self, num_episodes, success_rate, avg_reward, avg_length):
        test_log_file = os.path.join(self.log_dir, f"{self.algorithm}_test_results_stage{self.stage}_{self.current_time}.txt")
        with open(test_log_file, 'a') as f:
            f.write(f"Test Results (Stage {self.stage}):\n")
            f.write(f"Number of Episodes: {num_episodes}\n")
            f.write(f"Success Rate: {success_rate:.2f}\n")
            f.write(f"Average Reward: {avg_reward:.2f}\n")
            f.write(f"Average Episode Length: {avg_length:.2f}\n\n")

    def get_current_performance(self):
        if len(self.episode_rewards) == 0:                                                                      #TODO
            return 0, 0
        avg_reward = np.mean(self.episode_rewards[-100:])
        success_rate = np.mean(self.episode_successes[-100:])
        return avg_reward, success_rate