# StorageManager class designed to handle all file I/O operations related to saving and loading models
# replay buffers and metadata

import os
import torch
import pickle
import numpy as np


class StorageManager:
    def __init__(self, base_path, algorithm, stage):
        self.base_path = base_path
        self.algorithm = algorithm
        self.stage = stage
        self.model_dir = os.path.join(self.base_path, 'models', self.algorithm)
        self.ensure_directory_exists(self.model_dir)

    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def save_model(self, model, episode):                                                                           # using torch.save() to save model's state dictionary
        model_path = os.path.join(self.model_dir, f"{self.algorithm}_stage{self.stage}_episode{episode}")        # saving snapshots of model at different stages of training
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model, episode):                                                                           # restore previously saved model
        model_path = os.path.join(self.model_dir, f"{self.algorithm}_stage{self.stage}_episode{episode}")        # allows resuming training from previous point or load for evaluation
        if os.path.exists(model_path + "_actor"):
            model.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"No model found at {model_path}")
    
    def save_replay_buffer(self, replay_buffer, episode):                                                           # saving ReplayBuffer
        buffer_path = os.path.join(self.model_dir, f"replay_buffer_stage{self.stage}_episode{episode}.pkl")         # "experience replay"
        with open(buffer_path, 'wb') as f:
            pickle.dump(replay_buffer, f)                                                                           # "pickles" ReplayBuffer object and saves it to file
        print(f"Replay buffer saved to {buffer_path}")

    def load_replay_buffer(self, episode):
        buffer_path = os.path.join(self.model_dir, f"replay_buffer_stage{self.stage}_episode{episode}.pkl")         # loads previously saved ReplayBuffer 
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"No replay buffer found at {buffer_path}")
            return None
        
    def save_metadata(self, metadata, episode):                                                                     # metadata handling (like total timesteps, current episode, etc.) 
        metadata_path = os.path.join(self.model_dir, f"metadata_stage{self.stage}_episode{episode}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")

    def load_metadata(self, episode):
        metadata_path = os.path.join(self.model_dir, f"metadata_stage{self.stage}_episode{episode}.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"No metadata found at {metadata_path}")
            return None