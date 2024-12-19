import torch
import numpy as np
import argparse
import random
import yaml

from datasets import *
from model import *

RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

PROJECT_NAME = 'RUC_Grad_Design'

class DynamicClass:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
    args = DynamicClass(config)
    return args
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config.yaml")
args = parser.parse_args()