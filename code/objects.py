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

class play:
    def __init__(self, game_id, play_id, plays, tracking, ball_tracking, tackles, players):
        self.players = players
        self.play = plays.query("gameId == @game_id & playId ==  @play_id")
        self.ball_carry_id = self.play.ballCarrierId.reset_index(drop =1)[0]
        self.tracking_df = tracking.query("gameId == @game_id & playId ==  @play_id")
        self.ball_track = ball_tracking.query("gameId == @game_id & playId ==  @play_id")
        self.tackle_oppurtunities = tackles.query("gameId == @game_id & playId ==  @play_id")
        self.num_frames = max(self.tracking_df.frameId)
        self.eop = self.get_end_of_play_location()
        self.tracking_refined = {frame_id : self.refine_tracking(frame_id = frame_id) for frame_id in range(1, self.num_frames)}
        self.tracking_refined_stratified = {frame_id : {player_type : self.tracking_refined.get(frame_id).loc[(self.tracking_refined.get(frame_id)['type'] == player_type)] 
                                                        for player_type in ["Offense", "Defense", "Carrier"]} for frame_id in range(1, self.num_frames)}
    
    def get_end_of_play_location(self):
        end_of_play_carrier = self.tracking_df.query("nflId == @self.ball_carry_id & frameId == @self.num_frames")
        return end_of_play_carrier[["frameId", "x", "y"]].rename({"x" : "eop_x", "y" : "eop_y"}, axis = 1)
    
    def get_end_of_play_matrix(self, N):
        tackles_attempt_mat = np.zeros((int(120/N), math.ceil(54/N)))
        for item in list(zip(self.eop.eop_x, self.eop.eop_y)):
            tackles_attempt_mat[int(item[0]/N), int(item[1]/N)] = 1
        return tackles_attempt_mat
    
    def refine_tracking(self, frame_id):
        current_positions = self.tracking_df.query("frameId == @frame_id").merge(self.players, on = "nflId", how = "left")
        current_positions['type'] = current_positions['position'].apply(
            lambda x: "Offense" if x in ["QB", "TE", "WR", "G", "OLB", "RB", "C", "FB"] else "Defense")
        current_positions['type'] = current_positions.apply(lambda row: 'Ball' if pd.isna(row['nflId']) else row['type'], axis=1)
        current_positions.loc[current_positions.nflId == self.ball_carry_id, 'type'] = "Carrier"
        current_positions['dir_rad'] = np.radians(current_positions['dir']) # degrees point in y direction
        current_positions['Sx'] = current_positions['s'] * np.cos(current_positions['dir_rad'])
        current_positions['Sy'] = current_positions['s'] * np.sin(current_positions['dir_rad'])
        current_positions['Ax'] = current_positions['a'] * np.cos(current_positions['dir_rad'])
        current_positions['Ay'] = current_positions['a'] * np.sin(current_positions['dir_rad'])
        return current_positions[['nflId', 'x', 'y', 'Sx', 'Sy', 'Ax', 'Ay', 's', 'a', 'dis', 'o', 'dir', 'dir_rad', 'height', 'weight', 'type']]
    
    def get_grid_features(self, frame_id, N, matrix_form = True):
        stratified_dfs = self.tracking_refined_stratified[frame_id]
        grid_features = pd.DataFrame()
        return_mat = np.zeros((24, len(list(range(0, 120, N))), len(list(range(0, 54, N)))))
        for x_low in list(range(0, 120, N)):
            for y_low in list(range(0, 54, N)):
                off_df = stratified_dfs["Offense"]
                def_df = stratified_dfs["Defense"]
                ball_df = stratified_dfs["Carrier"]
                x_high = x_low + N
                y_high = y_low + N

                    # Extract relevant subsets of data
                off_subset = off_df[(off_df['x'] >= x_low) & (off_df['x'] < x_high) & (off_df['y'] >= y_low) & (off_df['y'] < y_high)]
                def_subset = def_df[(def_df['x'] >= x_low) & (def_df['x'] < x_high) & (def_df['y'] >= y_low) & (def_df['y'] < y_high)]
                ball_subset = ball_df[(ball_df['x'] >= x_low) & (ball_df['x'] < x_high) & (ball_df['y'] >= y_low) & (ball_df['y'] < y_high)]

                # Calculate statistics using vectorized operations
                current_offensive_player_density = len(off_subset)
                current_defensive_player_density = len(def_subset)
                current_ballcarrier_player_density = len(ball_subset)

                offense_directional_vector = np.cos(off_df['dir'] * (np.pi / 180)) * (x_low + N/2 - off_df['x']) + np.sin(off_df['dir'] * (math.pi / 180)) * (y_low + N/2 - off_df['y'])
                defense_directional_vector = np.cos(def_df['dir'] * (np.pi / 180)) * (x_low + N/2 - def_df['x']) + np.sin(def_df['dir'] * (math.pi / 180)) * (y_low + N/2 - def_df['y'])

                velocities_offensive_toward_point = off_df['s'] * offense_directional_vector
                velocities_defensive_toward_point = def_df['s'] * defense_directional_vector
                
                acceleration_offensive_toward_point = off_df['a'] * offense_directional_vector
                acceleration_defensive_toward_point = def_df['a'] * defense_directional_vector

                distance_offense_from_point = np.sqrt((off_df['x'] - (x_low + N/2))**2 + (off_df['y'] - (y_low + N/2))**2)
                distance_defensive_from_point = np.sqrt((def_df['x'] - (x_low + N/2))**2 + (def_df['y'] - (y_low + N/2))**2)

                velocities_ballcarrier_toward_point = ball_df['s'] * (np.cos(ball_df['dir'] * (math.pi / 180)) * (x_low + N/2 - ball_df['x']) +
                                                                    np.sin(ball_df['dir'] * (math.pi / 180)) * (y_low + N/2 - ball_df['y']))
                acceleration_ballcarrier_toward_point = ball_df['a'] * (np.cos(ball_df['dir'] * (math.pi / 180)) * (x_low + N/2 - ball_df['x']) +
                                                                        np.sin(ball_df['dir'] * (math.pi / 180)) * (y_low + N/2 - ball_df['y']))
                distance_ballcarrier_from_point = np.sqrt((ball_df['x'] - (x_low + N/2))**2 + (ball_df['y'] - (y_low + N/2))**2)
                ret = pd.DataFrame({'grid_id': [f"{x_low} {y_low}"],
                                                    'off_density': [current_offensive_player_density],
                                                    'def_density': [current_defensive_player_density],
                                                    'ballcarrier_density': [current_ballcarrier_player_density],
                                                    'off_velocity_mean': [np.mean(velocities_offensive_toward_point)],
                                                    'off_velocity_sum': [np.sum(velocities_offensive_toward_point)],
                                                    'off_velocity_std': [np.std(velocities_offensive_toward_point)],
                                                    'def_velocity_mean': [np.mean(velocities_defensive_toward_point)],
                                                    'def_velocity_sum': [np.sum(velocities_defensive_toward_point)],
                                                    'def_velocity_std': [np.std(velocities_defensive_toward_point)],
                                                    'ballcarrier_velocity': [velocities_ballcarrier_toward_point.values[0]],
                                                    'off_acc_mean': [np.mean(acceleration_offensive_toward_point)],
                                                    'off_acc_sum': [np.sum(acceleration_offensive_toward_point)],
                                                    'off_acc_std': [np.std(acceleration_offensive_toward_point)],
                                                    'def_acc_mean': [np.mean(acceleration_defensive_toward_point)],
                                                    'def_acc_sum': [np.sum(acceleration_defensive_toward_point)],
                                                    'def_acc_std': [np.std(acceleration_defensive_toward_point)],
                                                    'ballcarrier_acc': [acceleration_ballcarrier_toward_point.values[0]],
                                                    'off_distance_mean': [np.mean(distance_offense_from_point)],
                                                    'off_distance_sum': [np.sum(distance_offense_from_point)],
                                                    'off_distance_std': [np.std(distance_offense_from_point)],
                                                    'def_distance_mean': [np.mean(distance_defensive_from_point)],
                                                    'def_distance_sum': [np.sum(distance_defensive_from_point)],
                                                    'def_distance_std': [np.std(distance_defensive_from_point)],
                                                    'ballcarrier_distance': [distance_ballcarrier_from_point.values[0]]})
                if matrix_form:
                    return_mat[:, int(x_low/N), int(y_low/N)] = np.array(ret.drop(['grid_id'], axis = 1).iloc[0])
                else:
                    grid_features = pd.concat([grid_features, ret])
        if matrix_form:
            return return_mat
        else:
            return grid_features

    def get_all_grid_features(self, N):
        all_features = {}
        for frame_id in range(1, self.num_frames):
            all_features.update({frame_id : self.get_grid_features(frame_id = frame_id, N = N, matrix_form = True)})
        return(all_features)
    
    def predict_tackle_distribution(self, model):
        if len(self.tracking_refined.get(1).type.unique()) != 4:
            raise KeyError("None-Complete Tracking Data") # if not offense, defense, ball and carrier in play
        outputs = []
        pred_df = pd.DataFrame()
        for frame_id in range(1, self.num_frames):
            image = self.get_grid_features(frame_id = frame_id, N = model.N)
            image = image[None, :, :, :]
            output = model(torch.FloatTensor(image)).detach().numpy()
            for x in range(output.shape[1]):
                for y in range(output.shape[2]):
                    new_row = pd.DataFrame({"x" : [x*model.N+model.N/2], "y" : [y*model.N+model.N/2], "prob" : [output[0, x, y]], "frameId" : [frame_id]})
                    pred_df = pd.concat([pred_df, new_row], axis = 0)
        return(pred_df)
                


class TackleAttemptDataset:

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.num_samples = len(images)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        label = torch.FloatTensor(self.labels[idx])
        return image, label
    

class TackleNet(nn.Module):
    def __init__(self, N, nvar):
        super(TackleNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(nvar, 20, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(10, 5, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(5, 3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(math.ceil(120/N)*math.ceil(54/N)*3, 64)
        self.fc2 = nn.Linear(64, math.ceil(120/N)*math.ceil(54/N))
        self.N = N
        
    def forward(self, x):
        # Input shape: (batch_size, 24, 12, 6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        
        # Apply softmax to ensure the output sums to 1 along the channel dimension (12*6)
        x = x.view(-1, math.ceil(120/self.N), math.ceil(54/self.N))
        
        return x
    
def plot_predictions(prediction_output, true):
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))

    # Flatten the 8x8 grid of subplots to a 1D array for easier indexing
    axs = axs.flatten()

    # Loop through the 64 images and display each in a subplot
    for i in range(64):
        image = prediction_output[i].detach().numpy()
        true_image = true[i].detach().numpy()
        axs[i].imshow(image, cmap='Reds', interpolation='none')
        axs[i].axis('off')  # Turn off the axis for each subplot
        axs[i].set_title(f"Image {i + 1}")
        axs[i].grid()
        x_max = true_image.shape[0]
        y_max = true_image.shape[1]
        for k in range(x_max):
            for l in range(y_max):
                # axs[i].text(l, k, f'{image[k, l]:.1f}', ha='center', va='center', color='black')
                if true_image[k, l] == 1:
                    axs[i].plot(l, k, 'ro', markersize=5, color='blue')               
    # Adjust spacing between subplots to make them look better
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

class play_cache:
    def __init__(self, play_df):
        self.plays_zip = list(zip(play_df.gameId, play_df.playId))
        self.all_play = self.calculate_all_play_objects()

    def calculate_all_play_objects(self):
        print("Calculating all play_objects.")
        print("-----------------------------")
        all_plays = dict()
        for game_id, play_id in tqdm(self.plays_zip):
            all_plays[f"{game_id}_{play_id}"] = play(game_id, play_id)
        return all_plays 
    
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


