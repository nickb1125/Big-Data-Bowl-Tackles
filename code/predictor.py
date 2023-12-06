import pandas as pd
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
from objects import play, TackleAttemptDataset, TackleNet, plot_predictions, TackleNetEnsemble
import pickle

game_id = 2022100212
play_id = 2007

print("Loading base data")
print("-----------------")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])


print("Predicting")
print("-----------------")

# Get all defensive players to omit over play

play_object = play(game_id, play_id, tracking)
def_df = play_object.refine_tracking(frame_id = play_object.min_frame)["Defense"]
def_ids = def_df.nflId.unique()

# Define model

model = TackleNetEnsemble(num_models = 10, N = 5)

 # Predict original
all_pred = play_object.get_contribution_matricies(model = model, to_df = True)
all_pred.to_csv(f"{game_id}_{play_id}.csv")
    
