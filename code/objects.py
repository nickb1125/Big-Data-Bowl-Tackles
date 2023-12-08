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
import pickle

class play:
    def __init__(self, game_id, play_id, tracking):
        self.game_id = game_id
        self.play_id = play_id
        self.tracking_df = tracking.query("gameId == @game_id & playId ==  @play_id")
        self.ball_carry_id = self.tracking_df.ballCarrierId.reset_index(drop =1)[0]
        self.min_frame = min(self.tracking_df.frameId)
        self.num_frames = max(self.tracking_df.frameId)
        self.eop = self.get_end_of_play_location()
        self.yardlines = np.array(self.tracking_df.query("displayName == 'football'").x)

    def get_end_of_play_location(self):
        end_of_play_carrier = self.tracking_df.query("nflId == @self.ball_carry_id & frameId == @self.num_frames")
        return end_of_play_carrier[["frameId", "x", "y"]].rename({"x" : "eop_x", "y" : "eop_y"}, axis = 1)
    
    def get_end_of_play_matrix(self, N):
        tackles_attempt_mat = np.zeros((int(120/N), math.ceil(54/N)))
        for item in list(zip(self.eop.eop_x, self.eop.eop_y)):
            if item[0] >= 120:
                item = list(item)
                item[0] = 119
            if item[1] >= 54:
                item = list(item)
                item[1] = 53
            tackles_attempt_mat[int(item[0]/N), int(item[1]/N)] = 1
        return tackles_attempt_mat
    
    def refine_tracking(self, frame_id):
        this_frame = self.tracking_df.query("frameId == @frame_id")
        non_dict = this_frame[['nflId', 'x', 'y', 'Sx', 'Sy', 'Ax', 'Ay', 's', 'a', 'dis', 'o', 'dir', 'dir_rad', 'weight', 'type']]
        if len(non_dict.type.unique()) != 4:
            raise ValueError("Not does not account all player types")
        return {player_type : non_dict.loc[(non_dict['type'] == player_type)] 
                for player_type in ["Offense", "Defense", "Carrier"]}

    def get_grid_features(self, frame_id, N, plot = False, without_player_id = 0):

        stratified_dfs = self.refine_tracking(frame_id = frame_id)
        off_df = stratified_dfs["Offense"]
        def_df = stratified_dfs["Defense"]
        ball_df = stratified_dfs["Carrier"]
        if without_player_id != 0:
            def_df = def_df.query("nflId != @without_player_id")

        distance_offense_from_ballcarrier = np.sqrt((off_df['x'] - ball_df['x'].values[0])**2 + (off_df['y'] - ball_df['y'].values[0])**2)
        distance_defense_from_ballcarrier = np.sqrt((def_df['x'] - ball_df['x'].values[0])**2 + (def_df['y'] - ball_df['y'].values[0])**2)

        off_movement_features = get_player_movement_features(off_df, N)
        off_acc_mat = off_movement_features['field_weighted_acc']
        off_vel_mat = off_movement_features['field_weighted_velocity']
        # off_acc_mat_weight = np.array(distance_offense_from_ballcarrier)[:, np.newaxis, np.newaxis] * off_acc_mat
        # off_vel_mat_weight = np.array(distance_offense_from_ballcarrier)[:, np.newaxis, np.newaxis] * off_vel_mat

        def_movement_features = get_player_movement_features(def_df, N)
        def_acc_mat = def_movement_features['field_weighted_acc']
        def_vel_mat = def_movement_features['field_weighted_velocity']
        # def_acc_mat_weight = np.array(distance_defense_from_ballcarrier)[:, np.newaxis, np.newaxis] * def_acc_mat
        # def_vel_mat_weight = np.array(distance_defense_from_ballcarrier)[:, np.newaxis, np.newaxis] * def_vel_mat

        ball_movement_features = get_player_movement_features(ball_df, N)
        ball_acc_mat = ball_movement_features['field_weighted_acc']
        ball_vel_mat = ball_movement_features['field_weighted_velocity']

        off_density = get_player_field_densities(off_df, N)
        def_density = get_player_field_densities(def_df, N)

        ret = np.stack([off_density, def_density, 
                np.sum(ball_vel_mat, axis = 0), np.sum(ball_acc_mat, axis = 0),
                np.sum(off_vel_mat, axis = 0), np.std(off_vel_mat, axis = 0), 
                np.sum(off_acc_mat, axis = 0), np.std(off_acc_mat, axis = 0),
                np.sum(def_vel_mat, axis = 0), np.std(def_vel_mat, axis = 0), 
                np.sum(def_acc_mat, axis = 0), np.std(def_acc_mat, axis = 0),
                ])
        if not plot:
            return ret
        else:
            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(ret[i, :, :], cmap='viridis')
                ax.set_title(f'Dimension {i + 1}')

            # Adjust layout for better visualization
            plt.tight_layout()
            plt.show()

    def get_full_play_tackle_image(self, N, without_player_id = 0):
        batch_images = torch.FloatTensor(np.array([
            self.get_grid_features(frame_id=i, N=N, without_player_id=without_player_id)
            for i in range(self.min_frame, self.num_frames+1)]))
        return batch_images
    
    def predict_tackle_distribution(self, model, without_player_id = 0, to_df = True, marginal_x = False):
        batch_images = self.get_full_play_tackle_image(N = model.N, without_player_id=without_player_id)
        outputs_all = model.predict_pdf(batch_images)
        outputs = outputs_all['overall_pred'].detach().numpy()
        lower = outputs_all['lower'].detach().numpy()
        upper = outputs_all['upper'].detach().numpy()
        all_pred = outputs_all['all_pred'].detach().numpy()
        if marginal_x:
            outputs = np.sum(outputs, axis = 2)
            lower = np.sum(lower, axis = 2)
            upper = np.sum(upper, axis = 2)
            all_pred = np.sum(all_pred, axis = 3) # dim = (prediction_type (3), frame, x (24), y (11))
        if to_df:
            exp_ret = array_to_field_dataframe(input_array=outputs, N=model.N, marginalized=marginal_x)
            exp_ret["type"] = "prob"
            lower_ret = array_to_field_dataframe(input_array=lower, N=model.N, marginalized=marginal_x)
            lower_ret['type'] = "lower"
            upper_ret = array_to_field_dataframe(input_array=upper, N=model.N, marginalized=marginal_x)
            upper_ret['type'] = "upper"
            ret = pd.concat([exp_ret, lower_ret, upper_ret], axis = 0)
            ret['omit'] = without_player_id
            ret['frameId'] = ret['frameId']+self.min_frame
            ret = ret.pivot(index=['x', 'y', 'frameId', 'omit'], columns='type', values='prob').reset_index()
            return ret
        return outputs, lower, upper, all_pred
    
    def get_contribution_matricies(self, model, to_df = True, marginal_x = False):
        original = self.predict_tackle_distribution(model=model, without_player_id = 0, to_df = False, marginal_x = marginal_x)
        original_all_pred = original[3]

        # Predict ommissions
        def_df = self.refine_tracking(frame_id = self.min_frame)["Defense"]
        def_ids = def_df.nflId.unique()
        contribution_dict = dict()
        if to_df:
            original_df = self.predict_tackle_distribution(model=model, without_player_id = 0, to_df = True, marginal_x = marginal_x)
            contribution_list = [original_df]
        for id in tqdm(def_ids):
            w_omission = self.predict_tackle_distribution(model = model, without_player_id = id, to_df = False, marginal_x=marginal_x)
            w_omission_all_pred = w_omission[3]
            contribution_mat_all_pred = original_all_pred-w_omission_all_pred
            contribution_mat = np.mean(contribution_mat_all_pred, axis=0)
            contribution_mat_lower = np.min(contribution_mat_all_pred, axis=0)
            contribution_mat_upper = np.max(contribution_mat_all_pred, axis=0)
            if to_df:
                contribution_df_exp = array_to_field_dataframe(input_array=contribution_mat, N=model.N, marginalized = marginal_x)
                contribution_df_exp["type"] = "prob"
                contribution_df_lower = array_to_field_dataframe(input_array=contribution_mat_lower, N=model.N, marginalized = marginal_x)
                contribution_df_lower["type"] = "lower"
                contribution_df_upper = array_to_field_dataframe(input_array=contribution_mat_upper, N=model.N, marginalized = marginal_x)
                contribution_df_upper["type"] = "upper"
                contribution_df = pd.concat([contribution_df_exp, contribution_df_lower, contribution_df_upper], axis = 0)
                contribution_df['frameId'] = contribution_df['frameId']+self.min_frame+1
                contribution_df['omit'] = id
                contribution_df = contribution_df.pivot(index=['x', 'y', 'frameId', 'omit'], columns='type', values='prob').reset_index()
                contribution_list.append(contribution_df)

                continue
            contribution_dict.update({id : {"prob" : contribution_mat, "lower" : contribution_mat_lower, "upper" : contribution_mat_upper}})
        if to_df:
            return pd.concat(contribution_list, axis = 0)
        return contribution_dict
                


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
        self.conv1 = nn.Conv2d(nvar, 12, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 5, stride=1)        
        self.dropout1 = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(3168, math.ceil(120/N)*math.ceil(54/N))
        self.N = N
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = F.softmax(self.fc1(x), dim=1)
        
        # Apply softmax to ensure the output sums to 1 along the channel dimension (12*6)
        x = x.view(-1, math.ceil(120/self.N), math.ceil(54/self.N))
        
        return x
    
def get_player_movement_features(player_df, N):
    x_mat = np.tile(np.arange(0, 120, N)+(N/2), (math.ceil(54/N), 1))
    y_mat = np.transpose(np.tile(np.arange(0, 54, N)+(N/2), (math.ceil(120/N), 1)))
    distance_mat = np.zeros((player_df.shape[0], x_mat.shape[0], x_mat.shape[1]))
    once_weighted_velocity_mat = np.zeros((player_df.shape[0], x_mat.shape[0], x_mat.shape[1]))
    once_weighted_acceleration_mat = np.zeros((player_df.shape[0], x_mat.shape[0], x_mat.shape[1]))
    i = 0
    for index, row in player_df.iterrows():
        x_dist = x_mat - row.x
        y_dist = y_mat - row.y
        distance = np.sqrt((x_dist)**2 + (y_dist)**2)
        velocity_toward_grid = (row.Sx*x_dist + row.Sy*y_dist) / (distance+0.0001)
        acc_toward_grid = (row.Ax*x_dist + row.Ay*y_dist) / (distance+0.0001)
        weight_vel_by_dis_point_ball = velocity_toward_grid*(1/(distance+0.0001))
        weight_acc_by_dis_point_ball = acc_toward_grid*(1/(distance+0.0001))
        once_weighted_velocity_mat[i, :, :] = weight_vel_by_dis_point_ball
        once_weighted_acceleration_mat[i, :, :] = weight_acc_by_dis_point_ball
        distance_mat[i, :, :] = distance
        i += 1
    return {'distance' : distance_mat, 'field_weighted_velocity': once_weighted_velocity_mat, 'field_weighted_acc' : once_weighted_acceleration_mat}

def get_player_field_densities(player_df, N):
    density_mat = np.zeros((len(list(range(0, 54, N))), len(list(range(0, 120, N)))))
    for index, row in player_df.iterrows():
        x_rounded = math.floor(row.x / N)
        y_rounded = math.floor(row.y / N)
        try:
            density_mat[y_rounded, x_rounded] += 1
        except:
            print(y_rounded)
            print(x_rounded)
            raise ValueError
    return density_mat
    
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
                    axs[i].plot(l, k, 'ro', markersize=0.5, color='blue')               
    # Adjust spacing between subplots to make them look better
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

class TackleNetEnsemble:
    def __init__(self, num_models, N = 5):
        self.N = N
        self.models = dict()
        self.num_models = num_models
        for mod_num in range(num_models):
            with open(f"model_{mod_num}.pkl", 'rb') as f:
                model = pickle.load(f)
            self.models.update({mod_num : model})
    
    def predict_pdf(self, image):
        preds = []
        for mod_num in range(self.num_models):
            model = self.models[mod_num]
            pred = model(image)
            preds.append(pred)
        preds = torch.stack(preds)
        lower = torch.min(preds, axis = 0).values
        overall_pred = torch.mean(preds, axis = 0)
        upper = torch.max(preds, axis = 0).values
        return {'overall_pred' : overall_pred, 'lower' : lower, 'upper' : upper, 'all_pred' : preds}
    
def array_to_field_dataframe(input_array, N, marginalized = False):
    pred_list=[]
    if marginalized:
        for frame_id in range(input_array.shape[0]):
            for x in range(input_array.shape[1]):
                new_row = pd.DataFrame({"x" : [x*N+N/2], "prob" : [input_array[frame_id, x]], "frameId" : [frame_id]})
                pred_list.append(new_row)
        pred_df = pd.concat(pred_list, axis = 0)
        return pred_df 
    for frame_id in range(input_array.shape[0]):
        for x in range(input_array.shape[1]):
            for y in range(input_array.shape[2]):
                new_row = pd.DataFrame({"x" : [x*N+N/2], "y" : [y*N+N/2], "prob" : [input_array[frame_id, x, y]], "frameId" : [frame_id]})
                pred_list.append(new_row)
    pred_df = pd.concat(pred_list, axis = 0)
    return pred_df


                



        



