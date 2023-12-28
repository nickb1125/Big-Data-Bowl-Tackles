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
from objects import play, TackleAttemptDataset, plot_predictions
import pickle

print("Loading base data")
print("-----------------")

plays = pd.read_csv("data/nfl-big-data-bowl-2024/plays.csv")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])
 # Filter to only tracking plays meeting criteria
plays = tracking[['gameId', 'playId']].drop_duplicates().merge(plays, how = 'left', on = ['gameId', 'playId'])

load_test = True
N = 3
test_games = tracking.query("week == 9").gameId.unique()
test_plays = plays.query("(gameId in @test_games)")
train_val_games = tracking.query("week != 9").gameId.unique()
train_val_plays = plays.query("gameId in @train_val_games")

if load_test:
    print("Getting test images (all frames)....")
    print(f"(Using N = {N})")
    print("------------------------------------------")
    frame_dict = dict()
    for row in tqdm(range(test_plays.shape[0])):
        play_row = test_plays.iloc[row,]
        play_object = play(play_row.gameId, play_row.playId, tracking)
        for frame_id in range(play_object.min_frame, play_object.num_frames+1):
            frames_from_end = play_object.num_frames-frame_id
            if not frames_from_end in frame_dict.keys():
                frame_dict[frames_from_end] = {"images" : [], "labels" : [], "play_ids" : [], "frame_ids" : []}
            frame_dict[frames_from_end]['play_ids'].append(play_row.playId)
            frame_dict[frames_from_end]['frame_ids'].append(frame_id)
            try:
                image = play_object.get_grid_features(frame_id = frame_id, N = N)
            except ValueError:
                # print("Below is lacking a type of position and is being omitted, check if desired...")
                # print(row)
                continue # if not offense, defense, ball and carrier in play
            if np.isinf(image).any():
                # print("Below has infinity feature output and is being omitted, check if desired...")
                # print(row)
                continue
            frame_dict[frames_from_end]['images'].append(image)
            # frame_dict[frames_from_end]['labels'].append(play_object.get_end_of_play_matrix(N = N))
            frame_dict[frames_from_end]['labels'].append(play_object.eop[['eop_x', 'eop_y']].iloc[0].tolist())

    with open(f"data/test_frame_dict.pkl", f'wb') as outp:  # Overwrites any existing file.
        pickle.dump(frame_dict, outp, pickle.HIGHEST_PROTOCOL)
    
    print(f"Keys from end include: {frame_dict.keys()}")
    for frame_from_end in frame_dict.keys():
        tackle_dataset = TackleAttemptDataset(images = frame_dict[frame_from_end]['images'], 
                                              labels = frame_dict[frame_from_end]['labels'], 
                                              play_ids = frame_dict[frame_from_end]['play_ids'], 
                                              frame_ids = frame_dict[frame_from_end]['frame_ids'])
        with open(f"data/test_tackle_images_{frame_from_end}_from_end.pkl", f'wb') as outp:  # Overwrites any existing file.
            pickle.dump(tackle_dataset, outp, pickle.HIGHEST_PROTOCOL)

print("Getting training and validation images....")
print(f"(Using N = {N})")
print("------------------------------------------")

images = []
labels = []
play_ids = []
frame_ids = []
for bag in range(10):
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
            # print("Below is lacking a type of position and is being omitted, check if desired...")
            # print(row)
            continue # if not offense, defense, ball and carrier in play
        if np.isinf(image).any():
            # print("Below has infinity feature output and is being omitted, check if desired...")
            # print(row)
            continue
        images.append(image)
        # labels.append(play_object.get_end_of_play_matrix(N = N))
        labels.append(play_object.eop[['eop_x', 'eop_y']].iloc[0].tolist())
    tackle_dataset = TackleAttemptDataset(images = images, labels = labels, play_ids = play_ids, frame_ids = frame_ids)

    with open(f"data/tackle_image_bag_{bag}.pkl", f'wb') as outp:  # Overwrites any existing file.
        pickle.dump(tackle_dataset, outp, pickle.HIGHEST_PROTOCOL)





