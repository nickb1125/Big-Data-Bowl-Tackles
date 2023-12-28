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
from scipy.stats import multivariate_normal
from torchvision import transforms, utils, models
from torch.distributions import MultivariateNormal, OneHotCategorical
import torch.nn.init as init
import concurrent.futures


def plot_predictions(prediction_output, true):
    fig, axs = plt.subplots(5, 4, figsize=(16, 16))
    pi_models, normal_models = prediction_output[0], prediction_output[1]
    means = normal_models.mean # (Batch, Nmix, 2)
    cov = normal_models.covariance_matrix # (Batch, Nmix, 2, # (Batch, Nmix, 2))
    probs = pi_models.probs # (Batch, Nmix)

    x, y = torch.meshgrid(torch.linspace(0, 120, 120), torch.linspace(0, 53, 53))
    grid_points = torch.stack([x, y], dim=-1).view(-1, 2)

    # Flatten the 8x8 grid of subplots to a 1D array for easier indexing
    axs = axs.flatten()

    # Loop through the 64 images and display each in a subplot
    for i in range(20):
        # Calculate the mixture bivariate PDF
        pdf_values = np.array([(probs[i][k]*torch.exp(MultivariateNormal(means[i, k], cov[i,k]).log_prob(grid_points))).detach().numpy()
                                for k in range(means.shape[1])]) # (nmix, num_grid_points)        
        pdf = np.sum(pdf_values, axis = 0).reshape((120, 53))
        axs[i].imshow(pdf, cmap='Reds', interpolation='none')
        axs[i].axis('off')  # Turn off the axis for each subplot
        axs[i].set_title(f"Image {i + 1}")
        axs[i].grid()
        axs[i].plot(true[i, 1], true[i, 0], 'ro', markersize=0.5, color='blue')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

def plot_field(pdf_output, true):
    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    x, y = torch.meshgrid(torch.linspace(0, 120, 120), torch.linspace(0, 53, 53))
    grid_points = torch.stack([x, y], dim=-1).view(-1, 2)

    # Flatten the 8x8 grid of subplots to a 1D array for easier indexing
    axs = axs.flatten()

    # Loop through the 64 images and display each in a subplot
    for i in range(64):
        pdf = pdf_output[i, :, :]
        axs[i].imshow(pdf, cmap='Reds', interpolation='none')
        axs[i].axis('off')  # Turn off the axis for each subplot
        axs[i].set_title(f"Image {i + 1}")
        axs[i].grid()
        axs[i].plot(true[i, 1], true[i, 0], 'ro', markersize=0.5, color='blue')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


def calculate_expected_from_vec(marginalized_probs):
    field_array = np.array(range(0, 120, 1))+0.5
    expected_saved = sum(marginalized_probs*field_array)
    return expected_saved

def calculate_expected_from_mat(marginalized_probs_mat, for_metric = False):
    # marginalized_probs_mat.shape == (num_ensemble_models, relevant_frame_num, x_size)
    contribution_array = np.apply_along_axis(calculate_expected_from_vec, axis=2, arr=marginalized_probs_mat)
    # contribution_array.shape == (num_ensemble_models, relevant_frame_num)
    estimate = np.mean(contribution_array, axis=0)
    lower = np.percentile(contribution_array, q=2.5, axis=0)
    upper = np.percentile(contribution_array, q=97.5, axis=0)
    if for_metric:
        return {"estimate": -estimate, "lower": -upper, "upper": -lower}
    return {"estimate": estimate, "lower": lower, "upper": upper}

class play:
    def __init__(self, game_id, play_id, tracking):
        self.game_id = game_id
        self.play_id = play_id
        self.tracking_df = tracking.query("gameId == @game_id & playId ==  @play_id")
        self.playDirection = self.tracking_df.playDirection.reset_index(drop = 1)[0]
        self.ball_carry_id = self.tracking_df.ballCarrierId.reset_index(drop =1)[0]
        self.min_frame = min(self.tracking_df.frameId)
        self.num_frames = max(self.tracking_df.frameId)
        self.eop = self.get_end_of_play_location()
        self.yardlines = np.array(self.tracking_df.query("displayName == 'football'").sort_values(by='frameId').x)
        self.features_base_cache = dict()

    def get_end_of_play_location(self):
        end_of_play_carrier = self.tracking_df.query("nflId == @self.ball_carry_id & frameId == @self.num_frames")
        return end_of_play_carrier[["frameId", "x", "y"]].rename({"x" : "eop_x", "y" : "eop_y"}, axis = 1)
    
    def refine_tracking(self, frame_id):
        this_frame = self.tracking_df.query("frameId == @frame_id")
        non_dict = this_frame[['nflId', 'x', 'y', 'Sx', 'Sy', 'Ax', 'Ay', 's', 'a', 'dis', 'o', 'dir', 'dir_rad', 'weight', 'type']]
        if len(non_dict.type.unique()) != 4:
            raise ValueError("Not does not account all player types")
        return {player_type : non_dict.loc[(non_dict['type'] == player_type)] 
                for player_type in ["Offense", "Defense", "Carrier"]}

    def get_grid_features(self, frame_id, N, plot = False, without_player_id = 0):
        if self.features_base_cache == dict():
            stratified_dfs = self.refine_tracking(frame_id = frame_id)
            off_df = stratified_dfs["Offense"].reset_index(drop = 1)
            def_df = stratified_dfs["Defense"].reset_index(drop = 1)
            ball_df = stratified_dfs["Carrier"].reset_index(drop = 1)

            distance_offense_from_ballcarrier = np.sqrt((off_df['x'] - ball_df['x'].values[0])**2 + (off_df['y'] - ball_df['y'].values[0])**2).tolist()
            distance_defense_from_ballcarrier = np.sqrt((def_df['x'] - ball_df['x'].values[0])**2 + (def_df['y'] - ball_df['y'].values[0])**2).tolist()

            off_movement_features = get_player_movement_features(off_df, N)
            off_acc_mat = off_movement_features['field_weighted_acc']
            off_vel_mat = off_movement_features['field_weighted_velocity']
            #off_distance_mat = off_movement_features['distance']
            #off_acc_mat_weight = (1/np.array(distance_offense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * off_acc_mat)
            #off_vel_mat_weight = (1/np.array(distance_offense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * off_vel_mat)

            def_movement_features = get_player_movement_features(def_df, N)
            def_acc_mat = def_movement_features['field_weighted_acc']
            def_vel_mat = def_movement_features['field_weighted_velocity']
            #def_distance_mat = off_movement_features['distance']
            #def_acc_mat_weight = (1/np.array(distance_defense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * def_acc_mat)
            #def_vel_mat_weight = (1/np.array(distance_defense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * def_vel_mat)

            ball_movement_features = get_player_movement_features(ball_df, N)
            ball_acc_mat = ball_movement_features['field_weighted_acc']
            ball_vel_mat = ball_movement_features['field_weighted_velocity']

            off_density = get_player_field_densities(off_df, N)
            def_density = get_player_field_densities(def_df, N)

            self.features_base_cache.update({'off_density' : off_density, 'def_density' : def_density,
                                            'ball_vel_mat' : ball_vel_mat, 'ball_acc_mat' : ball_acc_mat,
                                            'off_vel_mat' : off_vel_mat, 'off_acc_mat' : off_acc_mat,
                                            'def_vel_mat' : def_vel_mat, 'def_acc_mat' : def_acc_mat,
                                            'distance_defense_from_ballcarrier' : distance_defense_from_ballcarrier,
                                            'distance_offense_from_ballcarrier' : distance_offense_from_ballcarrier,
                                            'def_df' : def_df})
        else:
            distance_offense_from_ballcarrier = self.features_base_cache['distance_offense_from_ballcarrier'].copy()
            distance_defense_from_ballcarrier = self.features_base_cache['distance_defense_from_ballcarrier'].copy()
            def_vel_mat = self.features_base_cache['def_vel_mat'].copy()
            def_acc_mat = self.features_base_cache['def_acc_mat'].copy()
            off_density = self.features_base_cache['off_density'].copy()
            def_density = self.features_base_cache['def_density'].copy()
            ball_vel_mat = self.features_base_cache['ball_vel_mat'].copy()
            ball_acc_mat = self.features_base_cache['ball_acc_mat'].copy()
            off_vel_mat = self.features_base_cache['off_vel_mat'].copy()
            off_acc_mat = self.features_base_cache['off_acc_mat'].copy()

        if without_player_id != 0:
            def_df = self.features_base_cache['def_df']
            player_index = def_df[def_df['nflId'] == without_player_id].index[0]
            distance_defense_from_ballcarrier = distance_defense_from_ballcarrier.copy()
            distance_defense_from_ballcarrier.pop(player_index)
            def_vel_mat = np.delete(def_vel_mat, player_index, axis = 0)
            def_acc_mat = np.delete(def_acc_mat, player_index, axis = 0)
        
        # Filter to 12 closest
        closest = np.argsort(distance_defense_from_ballcarrier)[:8]
        closest_off = np.argsort(distance_offense_from_ballcarrier)[:8]

        ret = np.stack([
                np.sum(off_density, axis=0), 
                np.sum(def_density, axis=0),
                np.sum(ball_vel_mat, axis = 0), 
                np.sum(ball_acc_mat, axis = 0),
                np.sum(off_vel_mat[closest_off], axis = 0), 
                np.sum(off_acc_mat[closest_off], axis = 0), 
                np.sum(def_vel_mat[closest], axis = 0),
                np.sum(def_acc_mat[closest], axis = 0),
                ])
        if not plot:
            return ret
        else:
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                img = ax.imshow(ret[i, :, :], cmap='viridis')
                ax.set_title(f'Dimension {i + 1}')
                img.set_clim(vmin=0, vmax=1)

            # Adjust layout for better visualization
            plt.tight_layout()
            plt.show()

            ret = ret[3:7, :, :]
            # Create a figure and 3D axis
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plot each image in the 3D stack
            for i in range(ret.shape[0]):
                x = np.arange(ret.shape[2])
                y = np.arange(ret.shape[1])
                X, Y = np.meshgrid(x, y)

                # Flatten the 2D arrays to 1D and use them as coordinates
                Z = np.ones_like(X) * i
                ax.plot_surface(X, Y, Z, facecolors=plt.cm.cool(ret[i]), rstride=1, cstride=1, alpha=0.5, antialiased=False)

            # Set labels and title
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=20, azim=45)

            plt.show()

    def get_full_play_tackle_image(self, N, without_player_id = 0):
        batch_images = torch.FloatTensor(np.array([
            self.get_grid_features(frame_id=i, N=N, without_player_id=without_player_id)
            for i in range(self.min_frame, self.num_frames+1)]))
        return batch_images
    
    def predict_tackle_distribution(self, model, without_player_id = 0):
        batch_images = self.get_full_play_tackle_image(N = model.N, without_player_id=without_player_id)
        outputs_all = model.predict_pdf(batch_images)
        return {'estimate' : outputs_all['overall_pred'], 'lower' : outputs_all['lower'], 'upper' : outputs_all['upper'], 
                'all' : outputs_all['all_pred']}
    
    def get_expected_eop(self, predict_object, return_all_model_results=False):
        expected_x_stratified = predict_object['all'] 
        num_mod, num_frame, rows, cols = np.indices(expected_x_stratified.shape)

        # Calculate the mean coordinates
        means_row = np.sum(rows * expected_x_stratified, axis = (2,3)) / np.sum(expected_x_stratified, axis = (2,3))
        if return_all_model_results:
            return means_row
        return np.percentile(means_row, axis = 0, q = 2.5), np.mean(means_row, axis = 0), np.percentile(means_row, axis = 0, q = 97.5)
    
    def get_expected_contribution(self, model, original_predict_object, w_omission_all_pred):
        # Get contribution:
        expected_omit = self.get_expected_eop(w_omission_all_pred, return_all_model_results=True)
        expected_orig = self.get_expected_eop(original_predict_object, return_all_model_results=True)
        contributions = expected_orig-expected_omit # num_mod, num_frame
        #if self.playDirection == 'left':
        #    contributions = -contributions
        #print(f"Expected Contributions: {np.mean(contributions, axis = 0)}")

        # -----------------------------------------
        
        # Get Sphere of Influence (number of square yards that are needed to account for 90% of the players influence)
        # Step 1: Flatten the array
        contribution_matrix = w_omission_all_pred['all'] - original_predict_object['all']
        contribution_matrix[contribution_matrix < 0] = 0 # filter to positive influence (since influence should sum so 0)
        flattened_array = contribution_matrix.reshape((model.num_models, self.num_frames-self.min_frame+1, -1))

        # Step 2: Sort the flattened array in descending order
        sorted_array = np.sort(flattened_array, axis=-1)[:, :, ::-1]

        # Step 3: Calculate the cumulative sum
        cumulative_sum = np.cumsum(sorted_array, axis=-1)

        # Step 4: Find the index where cumulative sum exceeds 90% of the total sum
        num_grid_points_90_percent = np.argmax(cumulative_sum >= 0.9 * np.sum(flattened_array, axis=-1)[:,:,np.newaxis], axis=-1)  + 1  # Add 1 because indexing starts from 0
        #print(f"Expected SOIs: {np.mean(num_grid_points_90_percent, axis = 0)}")
        #print("------------------------")
        return {"contribution" : (np.percentile(contributions, axis = 0, q=2.5), np.mean(contributions, axis = 0), np.percentile(contributions, axis = 0, q = 97.5)),
                "soi" : (np.percentile(num_grid_points_90_percent, axis = 0, q=2.5), np.mean(num_grid_points_90_percent, axis = 0), np.percentile(num_grid_points_90_percent, axis = 0, q = 97.5))}
    
    def get_plot_df(self, model):
        # Get df from original (i.e. no player replacement)
        original_no_omit = self.predict_tackle_distribution(model=model, without_player_id = 0)
        lower_eop_orig, exp_eop_orig, upper_eop_orig = self.get_expected_eop(predict_object=original_no_omit)
        outputs_no_omit, lower_no_omit, upper_no_omit = original_no_omit['estimate'], original_no_omit['lower'], original_no_omit['upper']
        exp_contribution, lower_contribution, upper_contribution = np.zeros(len(range(self.min_frame, self.num_frames+1))), np.zeros(len(range(self.min_frame, self.num_frames+1))), np.zeros(len(range(self.min_frame, self.num_frames+1)))
        exp_soi, lower_soi, upper_soi= np.zeros(len(range(self.min_frame, self.num_frames+1))), np.zeros(len(range(self.min_frame, self.num_frames+1))), np.zeros(len(range(self.min_frame, self.num_frames+1)))
        exp_ret = array_to_field_dataframe(input_array=outputs_no_omit, N=1, for_features=False)
        exp_ret["type"] = "prob"
        lower_ret = array_to_field_dataframe(input_array=lower_no_omit, N=1, for_features=False)
        lower_ret['type'] = "lower"
        upper_ret = array_to_field_dataframe(input_array=upper_no_omit, N=1, for_features=False)
        upper_ret['type'] = "upper"
        ret_tackle_probs = pd.concat([exp_ret, lower_ret, upper_ret], axis = 0)
        ret_tackle_probs['omit'] = 0
        ret_tackle_probs['frameId'] = ret_tackle_probs['frameId']+self.min_frame
        ret_tackle_probs = ret_tackle_probs.pivot(index=['x', 'y', 'frameId', 'omit'], columns='type', values='prob').reset_index()
        ret_contribution = pd.DataFrame({"frameId" : range(self.min_frame, self.num_frames+1), "exp_eop" : exp_eop_orig, 
                                         "exp_contribution" : exp_contribution, "lower_contribution" : lower_contribution, "upper_contribution" : upper_contribution,
                                         "exp_soi" : exp_soi, "lower_soi" : lower_soi, "upper_soi" : upper_soi})
        ret = ret_tackle_probs.merge(ret_contribution, how = "left", on = "frameId")

        output_list = [ret]
        # Predict ommissions
        def_df = self.refine_tracking(frame_id = self.min_frame)["Defense"]
        def_ids = def_df.nflId.unique()
        for id in tqdm(def_ids):
            print("--------------------------------------------")
            print(self.playDirection)
            print(f"nflId: {id}")
            original = self.predict_tackle_distribution(model=model, without_player_id = id)
            lower_eop, exp_eop, upper_eop = self.get_expected_eop(predict_object=original)
            print(f"Expected original EOP frame 1: {exp_eop_orig[0]} 95% CI: [{lower_eop_orig[0]}, {upper_eop_orig[0]}]")
            print(f"Expected with ommitted EOP frame 1: {exp_eop[0]} 95% CI: [{lower_eop[0]}, {upper_eop[0]}")
            metric_dict = self.get_expected_contribution(model=model, original_predict_object=original_no_omit, w_omission_all_pred=original)
            lower_contribution, exp_contribution, upper_contribution = metric_dict["contribution"]
            lower_soi, exp_soi, upper_soi = metric_dict["soi"]
            outputs, lower, upper = original['estimate']-outputs_no_omit, original['lower']-lower_no_omit, original['upper']-upper_no_omit
            print(f"Expected contribution frame 1: {exp_contribution[0]} 95% CI: [{lower_contribution[0]}, {upper_contribution[0]}]")
            print(f"Expected SOI frame 1: {exp_soi[0]} 95% CI: [{lower_soi[0]}, {upper_soi[0]}]")
            exp_ret = array_to_field_dataframe(input_array=outputs, N=1, for_features=False)
            exp_ret["type"] = "prob"
            lower_ret = array_to_field_dataframe(input_array=lower, N=1, for_features=False)
            lower_ret['type'] = "lower"
            upper_ret = array_to_field_dataframe(input_array=upper, N=1, for_features=False)
            upper_ret['type'] = "upper"
            ret_tackle_probs = pd.concat([exp_ret, lower_ret, upper_ret], axis = 0)
            ret_tackle_probs['omit'] = id
            ret_tackle_probs['frameId'] = ret_tackle_probs['frameId']+self.min_frame
            ret_tackle_probs = ret_tackle_probs.pivot(index=['x', 'y', 'frameId', 'omit'], columns='type', values='prob').reset_index()
            ret_contribution = pd.DataFrame({"frameId" : range(self.min_frame, self.num_frames+1), "exp_eop" : exp_eop_orig, 
                                         "exp_contribution" : exp_contribution, "lower_contribution" : lower_contribution, "upper_contribution" : upper_contribution,
                                         "exp_soi" : exp_soi, "lower_soi" : lower_soi, "upper_soi" : upper_soi})
            ret = ret_tackle_probs.merge(ret_contribution, how = "left", on = "frameId")
            output_list.append(ret)
            print("--------------------------------------------")
            
        return pd.concat(output_list, axis = 0)


class TackleAttemptDataset:
    def __init__(self, images, labels, play_ids, frame_ids):
        self.playIds = play_ids
        self.frameIds = frame_ids
        self.images = images
        self.labels = labels
        self.num_samples = len(labels)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        label = torch.FloatTensor(self.labels[idx])
        return image, label
    
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
        velocity_toward_grid[velocity_toward_grid < 0] = 0
        acc_toward_grid = (row.Ax*x_dist + row.Ay*y_dist) / (distance+0.0001)
        weight_vel_by_dis_point_ball = velocity_toward_grid*(1/(distance+0.0001))
        weight_acc_by_dis_point_ball = acc_toward_grid*(1/(distance+0.0001))
        once_weighted_velocity_mat[i, :, :] = weight_vel_by_dis_point_ball
        once_weighted_acceleration_mat[i, :, :] = weight_acc_by_dis_point_ball
        distance_mat[i, :, :] = distance
        i += 1
    return {'distance' : distance_mat, 'field_weighted_velocity': once_weighted_velocity_mat, 
            'field_weighted_acc' : once_weighted_acceleration_mat}

def get_player_field_densities(player_df, N):
    density_mat = np.zeros((player_df.shape[0], len(list(range(0, 54, N))), len(list(range(0, 120, N)))))
    for index, row in player_df.iterrows():
        x_rounded = math.floor(row.x / N)
        y_rounded = math.floor(row.y / N)
        try:
            density_mat[index, y_rounded, x_rounded] += 1
        except:
            print(player_df.shape[0])
            print(index)
            print(y_rounded)
            print(x_rounded)
            raise ValueError
    return density_mat
    
    
def array_to_field_dataframe(input_array, N, for_features=False):
    pred_list = []
    shape = input_array.shape
    if for_features:
        for frame_id in range(shape[0]):
            for feature_num in range(shape[1]):
                x_values = np.arange(0, shape[2]) * N + N/2
                y_values = np.arange(0, shape[3]) * N + N/2
                x, y = np.meshgrid(x_values, y_values, indexing='ij')
                new_rows = pd.DataFrame({
                    "x": x.flatten(),
                    "y": y.flatten(),
                    "feature_num": feature_num,
                    'value': input_array[frame_id, feature_num, :, :].flatten(),
                    "frameId": frame_id
                })
                pred_list.append(new_rows)
        pred_df = pd.concat(pred_list, axis=0)
    else:
        for frame_id in range(shape[0]):
            x_values = np.arange(0, shape[1]) * N + N/2
            y_values = np.arange(0, shape[2]) * N + N/2
            x, y = np.meshgrid(x_values, y_values, indexing='ij')
            new_rows = pd.DataFrame({
                "x": x.flatten(),
                "y": y.flatten(),
                "prob": input_array[frame_id, :, :].flatten(),
                "frameId": frame_id
            })
            pred_list.append(new_rows)
        pred_df = pd.concat(pred_list, axis=0)

    return pred_df

def get_discrete_pdf_from_mvn(model):
    x, y = np.meshgrid(np.linspace(0, 54, 54), np.linspace(0, 120, 120))
    coordinates = torch.Tensor(np.column_stack((x.ravel(), y.ravel())))
    pdf_values = np.exp(model.log_prob(coordinates).detach().numpy()).reshape(120, 54)
    if np.sum(pdf_values) < 0.0001:
        return np.zeros((120, 54))
    discrete_pdf = pdf_values / np.sum(pdf_values)

    return discrete_pdf

def get_mixture_pdf(normal_models, pi_models):
    mean = normal_models.mean # (Batch, Nmix, 2)
    cov = normal_models.covariance_matrix # (Batch, Nmix, 2, # (Batch, Nmix, 2))
    probs = pi_models.probs # (Batch, Nmix)
    pdfs = np.zeros((mean.shape[0], 120, 54))
    for batch in range(mean.shape[0]):
        pdf_values = np.array([probs[batch][k].detach().numpy()*get_discrete_pdf_from_mvn(MultivariateNormal(mean[batch, k], cov[batch,k]))
                        for k in range(mean.shape[1])]) # (nmix, num_grid_points)
        pdf = np.sum(pdf_values, axis = 0)
        pdfs[batch,:,:] = pdf
    return pdfs


class TackleNetEnsemble:
    def __init__(self, num_models, N, nmix):
        self.models = dict()
        self.N = N
        self.num_models = num_models
        for mod_num in range(1, num_models+1):
            model = BivariateGaussianMixture(nmix=nmix, full_cov=True)
            model.load_state_dict(torch.load(f'model_{mod_num}_weights.pth'))
            model.eval()
            self.models.update({mod_num : model})

    def predict_pdf(self, image):
        preds = []
        pis_list, mus_list, sigmas_list = [], [], []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Process models in parallel
            futures = [executor.submit(process_model, model, image) for model in self.models.values()]
            for future in concurrent.futures.as_completed(futures):
                pis, mus, sigmas, pred_pdf = future.result()
                pis_list.append(pis)
                mus_list.append(mus)
                sigmas_list.append(sigmas)
                preds.append(pred_pdf)

        pis_combined = np.stack(pis_list)
        mus_combined = np.stack(mus_list)
        sigmas_combined = np.stack(sigmas_list)

        # Reshape and compute new values
        pis_new = pis_combined.reshape((pis_combined.shape[1], -1)) / self.num_models
        mus_new = mus_combined.reshape((mus_combined.shape[1], -1, 2))
        sigmas_new = sigmas_combined.reshape((sigmas_combined.shape[1], -1, 2, 2))

        mixture_ensemble_object = (
            OneHotCategorical(probs=torch.Tensor(pis_new)),
            MultivariateNormal(loc=torch.Tensor(mus_new), covariance_matrix=torch.Tensor(sigmas_new))
        )

        preds = np.stack(preds)
        lower = np.percentile(preds, q=2.75, axis=0)
        overall_pred = np.mean(preds, axis=0)
        upper = np.percentile(preds, q=97.5, axis=0)

        return {'overall_pred': overall_pred, 'lower': lower, 'upper': upper,
                'all_pred': preds, 'mixture_return': mixture_ensemble_object}
    

class GaussianMixtureLoss(nn.Module):
    def __init__(self):
        super(GaussianMixtureLoss, self).__init__()

    def forward(self, output, y):
        pi_model, normal_model = output[0], output[1]
        loglik = normal_model.log_prob(y.unsqueeze(1).expand_as(normal_model.loc))
        loss = -torch.logsumexp(torch.log(pi_model.probs) + loglik, dim=1)
        return loss.sum()

                
class BivariateGaussianMixture(nn.Module):
    def __init__(self, nmix, full_cov=True):
        super().__init__()
        self.nmix = nmix
        self.conv1 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride=1) 
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride=1)  
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride=1)          
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.mu_net = nn.Sequential(
                            nn.Linear(504, 50),
                            nn.ReLU(),
                            nn.Linear(50, 10),
                            nn.ReLU(),
                            nn.Linear(10, nmix * 2),
                        )
        self.cov_net = nn.Sequential(
                            nn.Linear(504, 50),
                            nn.ReLU(),
                            nn.Linear(50, 10),
                            nn.ReLU(),
                            nn.Linear(10, nmix * 2),
                        )
        self.pi_net = nn.Sequential(
                            nn.Linear(504, 50),
                            nn.ReLU(),
                            nn.Linear(50, 10),
                            nn.ReLU(),
                            nn.Linear(10, nmix),
                        )
        self.elu = nn.ELU()
        init.constant_(self.mu_net[-1].bias, 0) # sigmoid(0) = 0.5
        init.constant_(self.cov_net[-1].bias, 10) # Start variance slow and let modes expand 
        # init.constant_(self.pi_net[-1].bias, 150)

    def forward(self, x):
        #print(f"OG Shape: {x.shape}")
        x = self.conv1(x)
        #print(f"After Conv Shape: {x.shape}")
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        #print(f"After Flatten Shape: {x.shape}")
        
        # print(f"After Encoder Shape: {x.shape}")
        field_scaler = torch.stack([torch.Tensor([53.3, 120])] * self.nmix, dim=0)
        mean = torch.sigmoid(self.mu_net(x))
        mean = mean.reshape(mean.shape[0], self.nmix, 2)*field_scaler
        # print(f"Mean Shape: {mean.shape}")
        # print(f"Means: {mean[0]}")
        cov = nn.ELU()(self.cov_net(x))+1+1e-6
        # cov = torch.sigmoid(self.cov_net(x))
        cov = cov.reshape(mean.shape[0], self.nmix, 2)
        cov = torch.diag_embed(cov)
        # print(f"Cov Shape: {cov.shape}")
        # print(f"Cov: {cov[0]}")
        params = F.softmax(self.pi_net(x), dim = 1)
        # print(f"Mixture Shape: {params.shape}")
        # print(f"Mixture Probs: {params[0]}")
        return OneHotCategorical(probs=params), MultivariateNormal(loc=mean, covariance_matrix=cov)
    
def process_model(model, image):
    pred = model(image)
    pis = pred[0].probs.detach().numpy()
    mus = pred[1].mean.detach().numpy()
    sigmas = pred[1].covariance_matrix.detach().numpy()
    pred_pdf = get_mixture_pdf(normal_models=pred[1], pi_models=pred[0])
    return pis, mus, sigmas, pred_pdf
    
def plot_random_ensemble_predictions_all(tracking, model):
    random_instant = tracking[['gameId', 'playId', 'frameId']].drop_duplicates().sample(n=1)
    frame_id = random_instant.frameId.reset_index(drop = 1)[0]
    game_id = random_instant.gameId.reset_index(drop = 1)[0]
    play_id = random_instant.playId.reset_index(drop = 1)[0]
    print(f"GameId : {game_id}")
    print(f"PlayId : {play_id}")
    print(f"FrameId : {frame_id}")

    play_object = play(game_id, play_id, tracking)
    true_x = play_object.eop['eop_x'].reset_index(drop=1)[0]
    true_y = play_object.eop['eop_y'].reset_index(drop=1)[0]
    frame_id = frame_id - play_object.min_frame

    # Plot all models
    pred = play_object.predict_tackle_distribution(model, 0)
    data = pred['all'][:, frame_id, :, :]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    for i in range(10):
        axes[i].imshow(data[i, :, :], cmap='Reds')  # Change cmap as needed
        axes[i].axis('off')  # Turn off axis labels
        axes[i].set_title(f'Model {i + 1}')
        axes[i].plot(true_y, true_x, 'ro', markersize=1, color='blue')
    plt.tight_layout()
    plt.show()

    # Plot ensemble
    data = pred['estimate'][frame_id, :, :]
    plt.imshow(data, cmap='Reds')  # Change cmap as needed
    plt.plot(true_y, true_x, 'ro', markersize=1, color='blue')
    plt.axis('off')  # Turn off axis labels
    plt.title('Ensemble')
    plt.show()




        






