import logging
from datetime import datetime

class DataLogger:
    def __init__(self, log_file=None):
        if log_file is None:
            log_file = f"vrx_drl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
    def log_step(self, episode, step, state, action, reward, next_state, done):
        logging.info(f"Episode: {episode}, Step: {step}, State: {state}, "
                     f"Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
    
    def log_episode(self, episode, total_reward, avg_reward):
        logging.info(f"Episode {episode} finished. Total Reward: {total_reward}, Average Reward: {avg_reward}")
    
    def log_training(self, episode, loss, q_value):
        logging.info(f"Training - Episode: {episode}, Loss: {loss}, Q-value: {q_value}")