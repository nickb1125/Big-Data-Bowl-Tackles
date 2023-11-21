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
            if item[0] > 120:
                item = list(item)
                item[0] = 120
            if item[1] > 54:
                item = list(item)
                item[1] = 54
            tackles_attempt_mat[int(item[0]/N), int(item[1]/N)] = 1
        return tackles_attempt_mat
    
    def refine_tracking(self, frame_id):
        current_positions = self.tracking_df.query("frameId == @frame_id").merge(self.players, on = "nflId", how = "left")
        current_positions['type'] = current_positions['position'].apply(
            lambda x: "Offense" if x in ["QB", "TE", "WR", "G", "OLB", "RB", "C", "FB"] else "Defense")
        current_positions['type'] = current_positions.apply(lambda row: 'Ball' if pd.isna(row['nflId']) else row['type'], axis=1)
        current_positions.loc[current_positions.nflId == self.ball_carry_id, 'type'] = "Carrier"
        current_positions['dir_rad'] = np.radians(current_positions['dir']) # fix degrees
        current_positions['Sx'] = current_positions['s'] * np.cos(current_positions['dir_rad'])
        current_positions['Sy'] = current_positions['s'] * np.sin(current_positions['dir_rad'])
        current_positions['Ax'] = current_positions['a'] * np.cos(current_positions['dir_rad'])
        current_positions['Ay'] = current_positions['a'] * np.sin(current_positions['dir_rad'])
        current_positions['x'] = current_positions['x'].apply(lambda value: max(0, min(119.9, value)))
        current_positions['y'] = current_positions['y'].apply(lambda value: max(0, min(53.9, value)))
        return current_positions[['nflId', 'x', 'y', 'Sx', 'Sy', 'Ax', 'Ay', 's', 'a', 'dis', 'o', 'dir', 'dir_rad', 'height', 'weight', 'type']]

    def get_grid_features_simple(self, frame_id, N):
        return_mat = np.zeros((3, len(list(range(0, 120, N))), len(list(range(0, 54, N)))))
        stratified_dfs = self.tracking_refined_stratified[frame_id]
        off_df = stratified_dfs["Offense"]
        def_df = stratified_dfs["Defense"]
        ball_df = stratified_dfs["Carrier"]
        for _, row in off_df.iterrows():
            x, y = row['x'], row['y']
            return_mat[0, int(x / N), int((54 - y) / N)] += 1
        for _, row in def_df.iterrows():
            x, y = row['x'], row['y']
            return_mat[1, int(x / N), int((54 - y) / N)] += 1
        for _, row in ball_df.iterrows():
            x, y = row['x'], row['y']
            return_mat[2, int(x / N), int((54 - y) / N)] += 1
        return return_mat

    def get_grid_features(self, frame_id, N, matrix_form = True, plot = None, without_player_id = 0):
        stratified_dfs = self.tracking_refined_stratified[frame_id]
        grid_features = pd.DataFrame()
        return_mat = np.zeros((16, len(list(range(0, 120, N))), len(list(range(0, 54, N)))))
        off_df = stratified_dfs["Offense"]
        def_df = stratified_dfs["Defense"]
        if without_player_id != 0:
            def_df = def_df.query("nflId != @without_player_id")
        ball_df = stratified_dfs["Carrier"]
        current_x = int(ball_df.x.values[0])
        for x_low in list(range(0, 120, N)):
            for y_low in list(range(0, 54, N)):
                x_high, x_mid = x_low + N, x_low + (N/2)
                y_high, y_mid = y_low + N, y_low + (N/2)

                # Extract relevant subsets of data
                off_subset = off_df.loc[(off_df['x'] >= x_low) & (off_df['x'] < x_high) & (off_df['y'] >= y_low) & (off_df['y'] < y_high)]
                def_subset = def_df.loc[(def_df['x'] >= x_low) & (def_df['x'] < x_high) & (def_df['y'] >= y_low) & (def_df['y'] < y_high)]
                ball_subset = ball_df.loc[(ball_df['x'] >= x_low) & (ball_df['x'] < x_high) & (ball_df['y'] >= y_low) & (ball_df['y'] < y_high)]

                # Distance
                distance_offense_from_ballcarrier = np.sqrt((off_df['x'] - ball_df['x'].values[0])**2 + (off_df['y'] - ball_df['y'].values[0])**2)
                distance_defense_from_ballcarrier = np.sqrt((def_df['x'] - ball_df['x'].values[0])**2 + (def_df['y'] - ball_df['y'].values[0])**2)
                distance_offense_from_point = np.sqrt((off_df['x'] - (x_low + N/2))**2 + (off_df['y'] - (y_low + N/2))**2)
                distance_defense_from_point = np.sqrt((def_df['x'] - (x_low + N/2))**2 + (def_df['y'] - (y_low + N/2))**2)
                distance_ballcarrier_from_point = np.sqrt((ball_df['x'] - (x_low + N/2))**2 + (ball_df['y'] - (y_low + N/2))**2)


                # Calculate x,y differences from point
                dx_off, dy_off = off_df['x'] - x_mid, off_df['y'] - y_mid
                dx_def, dy_def = def_df['x'] - x_mid, def_df['y'] - y_mid
                dx_bc, dy_bc = ball_df['x'] - x_mid, ball_df['y'] - y_mid

                # Calculate dot product of change and Sx, Sy, Ax, Ay
                speed_dot_off = off_df['Sx']*dx_off + off_df['Sy']*dy_off
                speed_dot_def = def_df['Sx']*dx_def + def_df['Sy']*dy_def
                speed_dot_bc = ball_df['Sx']*dx_bc + ball_df['Sy']*dy_bc
                acc_dot_off = off_df['Ax']*dx_off + off_df['Ay']*dy_off
                acc_dot_def = def_df['Ax']*dx_def + def_df['Ay']*dy_def
                acc_dot_bc = ball_df['Ax']*dx_bc + ball_df['Ay']*dy_bc

                # Velocity toward grid point 
                off_velocity_toward_grid = speed_dot_off / (distance_offense_from_point+0.0001)
                def_velocity_toward_grid = speed_dot_def / (distance_defense_from_point+0.0001)
                ballcarrier_velocity_toward_grid = speed_dot_bc / (distance_ballcarrier_from_point+0.0001)

                # Acceleration toward grid point
                off_acc_toward_grid = acc_dot_off / (distance_offense_from_point+0.0001)
                def_acc_toward_grid = acc_dot_def / (distance_defense_from_point+0.0001)
                ballcarrier_acc_toward_grid = acc_dot_bc / (distance_ballcarrier_from_point+0.0001)

                # Weighted 
                off_weight_vel_by_dis_point = off_velocity_toward_grid*(1/distance_offense_from_point+0.0001)
                def_weight_vel_by_dis_point = def_velocity_toward_grid*(1/distance_defense_from_point+0.0001)
                off_weight_vel_by_dis_point_ball = off_weight_vel_by_dis_point*(1/distance_offense_from_ballcarrier+0.0001)
                def_weight_vel_by_dis_point_ball = def_weight_vel_by_dis_point*(1/distance_defense_from_ballcarrier+0.0001)
                ball_weight_vel_by_dis_point = ballcarrier_velocity_toward_grid*(1/distance_ballcarrier_from_point+0.0001)
                
                off_weight_acc_by_dis_point = off_acc_toward_grid*(1/distance_offense_from_point+0.0001)
                def_weight_acc_by_dis_point = def_acc_toward_grid*(1/distance_defense_from_point+0.0001)
                off_weight_acc_by_dis_point_ball = off_weight_acc_by_dis_point*(1/distance_offense_from_ballcarrier+0.0001)
                def_weight_acc_by_dis_point_ball = def_weight_acc_by_dis_point*(1/distance_defense_from_ballcarrier+0.0001)
                ball_weight_acc_by_dis_point = ballcarrier_acc_toward_grid*(1/distance_ballcarrier_from_point+0.0001)

                off_weights = off_subset['weight']
                def_weights = def_subset['weight']
                
                if matrix_form:
                    ret = [np.sum(off_weights), np.sum(def_weights), np.sum(off_weight_vel_by_dis_point),
                           np.std(off_weight_vel_by_dis_point), np.sum(def_weight_vel_by_dis_point), np.std(def_weight_vel_by_dis_point),
                           ball_weight_vel_by_dis_point.values[0], np.sum(off_weight_vel_by_dis_point_ball), np.sum(def_weight_vel_by_dis_point_ball),
                           np.sum(off_weight_acc_by_dis_point), np.std(off_weight_acc_by_dis_point), np.sum(def_weight_acc_by_dis_point),
                           np.std(def_weight_acc_by_dis_point), np.sum(off_weight_acc_by_dis_point_ball), np.sum(def_weight_acc_by_dis_point_ball),
                           ball_weight_acc_by_dis_point.values[0]]
                    return_mat[:, int(x_low/N), int((54-y_low)/N)] = ret # flipped so that (1,1) is bottom corner
                if (not matrix_form) or (plot) :
                    ret = pd.DataFrame({'x': x_low+(N/2), 'y' : y_low+(N/2),
                                    "weighted_off_grid" : [np.sum(off_weights)],
                                                    "weighted_def_grid" : [np.sum(def_weights)],
                                                    'off_weight_vel_by_dis_point': [np.sum(off_weight_vel_by_dis_point)],
                                                    'off_weight_vel_by_dis_point_sd': [np.std(off_weight_vel_by_dis_point)],
                                                    'def_weight_vel_by_dis_point': [np.sum(def_weight_vel_by_dis_point)],
                                                    'def_weight_vel_by_dis_point_sd': [np.std(def_weight_vel_by_dis_point)],
                                                    'ball_weight_vel_by_dis_point': [ball_weight_vel_by_dis_point.values[0]],
                                                    'off_weight_vel_by_dis_point_ball': [np.sum(off_weight_vel_by_dis_point_ball)],
                                                    'def_weight_vel_by_dis_point_ball': [np.sum(def_weight_vel_by_dis_point_ball)],
                                                    'off_weight_acc_by_dis_point' : [np.sum(off_weight_acc_by_dis_point)],
                                                    'off_weight_sd_by_dis_point_sd' : [np.std(off_weight_acc_by_dis_point)],
                                                    'def_weight_acc_by_dis_point' : [np.sum(def_weight_acc_by_dis_point)],
                                                    'def_weight_sd_by_dis_point_sd' : [np.std(def_weight_acc_by_dis_point)],
                                                    'off_weight_acc_by_dis_point_ball': [np.sum(off_weight_acc_by_dis_point_ball)],
                                                    'def_weight_acc_by_dis_point_ball' : [np.sum(def_weight_acc_by_dis_point_ball)],
                                                    'ball_weight_acc_by_dis_point' : [ball_weight_acc_by_dis_point.values[0]]
                                                    })
                    grid_features = pd.concat([grid_features, ret])
        if plot:
            labels = ret.columns[2:].tolist()
            fig, axs = plt.subplots(4, 4, figsize=(16, 16))
            axs = axs.flatten()
            for i in range(16):
                image = return_mat[i, :, :].T
                axs[i].imshow(image, cmap='Blues', interpolation='none')
                axs[i].axis('off')  # Turn off the axis for each subplot
                axs[i].set_title(labels[i])
                axs[i].grid()             
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.show()
        if matrix_form:
            return return_mat 
        else:
            return grid_features

    def get_all_grid_features_for_plot(self, N):
        all_features = pd.DataFrame()
        for frame_id in range(1, self.num_frames):
            grid_feat = self.get_grid_features(frame_id = frame_id, N = N, matrix_form = False)
            grid_feat['frameId'] = frame_id
            all_features = pd.concat([all_features, grid_feat], axis = 0)
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


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted_matrix, true_matrix):
        # Find the index of the maximum predicted probability for each sample
        _, predicted_indices = predicted_matrix.max(dim=1)

        # Create one-hot encoded tensor from the predicted indices
        predicted_one_hot = torch.zeros_like(predicted_matrix)
        predicted_one_hot.scatter_(1, predicted_indices.unsqueeze(1), 1)

        # Calculate the L1 distance between the predicted one-hot and true matrix
        loss = nn.functional.l1_loss(predicted_one_hot, true_matrix)

        return loss
                


class TackleAttemptDataset:

    def __init__(self, images, labels, play_ids, frame_ids):
        self.playIds = play_ids
        self.frameIds = frame_ids
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
        self.conv1 = nn.Conv2d(nvar, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(7200, math.ceil(120/N)*math.ceil(54/N))
        self.N = N
        
    def forward(self, x):
        # Input shape: (batch_size, 24, 12, 6)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = F.softmax(self.fc1(x), dim=1)
        
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
    
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


