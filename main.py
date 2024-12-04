import torch
import warnings
warnings.filterwarnings('ignore')

from model import T1PlatformTransformerEncoder, T1ObjectTransformerDecoder, OPT1, OPT1Noisy
from utils import batchestize, unbatchestize
from env import PuzzleEnv
from agent import Agent

import random
from time import sleep
from torch import optim
import matplotlib.pyplot as plt

from torch import nn



env = PuzzleEnv()

policy = OPT1Noisy(temperature=0.0001)

POLICY_SAVE = "policy_d_82000"

policy.load_state_dict(torch.load(f"./model_saves/{POLICY_SAVE}.pt"))

agent = Agent(policy=policy)

agent.train(100_000_000, env)

# env = PuzzleEnv()

