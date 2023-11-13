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

random.seed(2)

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

N = 5
print("Getting training and validation images....")
print(f"(Using N = {N})")
print("------------------------------------------")

# Choose random frame from each play

images = []
labels = []
for row in tqdm(plays.playId):
    play_row = plays.iloc[row,]
    play_object = play(play_row.gameId, play_row.playId, plays, tracking, ball_tracking, tackles, players)
    frame_id = random.randint(1, play_object.num_frames)
    if play_object.num_frames <= frame_id:
        continue # if not n frames happened
    if len(play_object.tracking_refined.get(1).type.unique()) != 4:
        print("Below is lacking a type of position and is being omitted, check if desired...")
        print(row)
        continue # if not offense, defense, ball and carrier in play
    image = play_object.get_grid_features(frame_id = frame_id, N = N)
    if np.isinf(image).any():
        print("Below has infinity feature output and is being omitted, check if desired...")
        print(row)
        continue
    images.append(image)
    labels.append(play_object.get_end_of_play_matrix(N = N))
tackle_dataset = TackleAttemptDataset(images = images, labels = labels)

with open("data/tackle_images_5.pkl", f'wb') as outp:  # Overwrites any existing file.
    pickle.dump(tackle_dataset, outp, pickle.HIGHEST_PROTOCOL)

