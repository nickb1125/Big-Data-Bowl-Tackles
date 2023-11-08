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
from objects import euclidean_distance, play, TackleAttemptDataset, TackleNet, plot_predictions, play_cache
import pickle

print("Loading base data")
print("-----------------")

# Read the CSV files
games = pd.read_csv("data/nfl-big-data-bowl-2024/games.csv")
players = pd.read_csv("data/nfl-big-data-bowl-2024/players.csv")

# Calculate height in inches
players['height'] = players['height'].str.extract(r'(\d+)').astype(int) * 12 + players['height'].str.extract(r'-(\d+)').astype(int)

# Select columns
players = players[['displayName', 'nflId', 'height', 'weight', 'position']]

plays = pd.read_csv("data/nfl-big-data-bowl-2024/plays.csv")
tackles = pd.read_csv("data/nfl-big-data-bowl-2024/tackles.csv")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_week_{week}.csv") for week in range(1, 10)])
ball_tracking = tracking.loc[tracking['nflId'].isna()][["gameId", "frameId", "playId", "x", "y"]].rename({"x" : "ball_x", "y" : "ball_y"}, axis = 1)


print("Predicting")
print("-----------------")

# Get model

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict

game_id = 2022100908
play_id = 3537

play_object = play(game_id, play_id, plays, tracking, ball_tracking, tackles, players)
play_object.predict_tackle_distribution(model = model).to_csv(f"{game_id}_{play_id}.csv")