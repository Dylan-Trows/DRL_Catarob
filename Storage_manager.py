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
        