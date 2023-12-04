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
from objects import euclidean_distance, play, TackleAttemptDataset, TackleNet, plot_predictions
import pickle

print("Loading base data")
print("-----------------")

plays = pd.read_csv("data/nfl-big-data-bowl-2024/plays.csv")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])
 # Filter to only tracking plays meeting criteria
plays = tracking[['gameId', 'playId']].drop_duplicates().merge(plays, how = 'left', on = ['gameId', 'playId'])

load_test = False
N = 1
test_games = tracking.query("week == 9").gameId.unique()
test_plays = plays.query("(gameId in @test_games)")
train_val_games = tracking.query("week != 9").gameId.unique()
train_val_plays = plays.query("gameId in @train_val_games")

if load_test:
    print("Getting test images (all frames)....")
    print(f"(Using N = {N})")
    print("------------------------------------------")

    images = []
    labels = []
    play_ids = []
    frame_ids = []
    for row in tqdm(range(test_plays.shape[0])):
        play_row = test_plays.iloc[row,]
        play_object = play(play_row.gameId, play_row.playId, plays, tracking, ball_tracking, tackles, players)
        frame_id = random.randint(1, play_object.num_frames)
        # for frame_id in range(1, play_object.num_frames):
        play_ids.append(play_row.playId)
        frame_ids.append(frame_id)
        try:
            image = play_object.get_grid_features(frame_id = frame_id, N = N)
        except ValueError:
            print("Below is lacking a type of position and is being omitted, check if desired...")
            print(row)
            continue # if not offense, defense, ball and carrier in play
        if np.isinf(image).any():
            print("Below has infinity feature output and is being omitted, check if desired...")
            print(row)
            continue
        images.append(image)
        labels.append(play_object.get_end_of_play_matrix(N = N))
    tackle_dataset = TackleAttemptDataset(images = images, labels = labels, play_ids = play_ids, frame_ids = frame_ids)

    with open("data/tackle_images_10_output_5_test.pkl", f'wb') as outp:  # Overwrites any existing file.
        pickle.dump(tackle_dataset, outp, pickle.HIGHEST_PROTOCOL)

print("Getting training and validation images....")
print(f"(Using N = {N})")
print("------------------------------------------")

images = []
labels = []
play_ids = []
frame_ids = []
for bag in range(1):
    print(f"COMPLETING BAG {bag}....")
    # Choose random frame from each play
    images = []
    labels = []
    for row in tqdm(range(train_val_plays.shape[0])):
        play_row = train_val_plays.iloc[row,]
        play_object = play(play_row.gameId, play_row.playId, tracking)
        frame_id = random.randint(play_object.min_frame, play_object.num_frames)
        play_ids.append(play_row.playId)
        frame_ids.append(frame_id)
        try:
            image = play_object.get_grid_features(frame_id = frame_id, N = N)
        except ValueError:
            print("Below is lacking a type of position and is being omitted, check if desired...")
            print(row)
            continue # if not offense, defense, ball and carrier in play
        if np.isinf(image).any():
            print("Below has infinity feature output and is being omitted, check if desired...")
            print(row)
            continue
        images.append(image)
        labels.append(play_object.get_end_of_play_matrix(N = N))
    tackle_dataset = TackleAttemptDataset(images = images, labels = labels, play_ids = play_ids, frame_ids = frame_ids)

    with open(f"data/tackle_images_1_output_1_bag_{bag}_stratified.pkl", f'wb') as outp:  # Overwrites any existing file.
        pickle.dump(tackle_dataset, outp, pickle.HIGHEST_PROTOCOL)





