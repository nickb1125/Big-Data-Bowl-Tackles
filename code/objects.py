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
        pdf_values = np.array([probs[i][k]*np.exp(MultivariateNormal(means[i, k], cov[i,k]).log_prob(grid_points).detach().numpy())
                                for k in range(means.shape[1])]) # (nmix, num_grid_points)
        pdf = np.sum(pdf_values, axis = 0).view(120, 53)
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
        # Filter to 7 closest so that omitting a player will not bias results
        closest = np.argsort(distance_defense_from_ballcarrier)[:11]
        closest_off = np.argsort(distance_offense_from_ballcarrier)[:11]

        off_movement_features = get_player_movement_features(off_df, N)
        # off_acc_mat = off_movement_features['field_weighted_acc']
        off_vel_mat = off_movement_features['field_weighted_velocity'][closest_off]
        # off_distance_mat = off_movement_features['distance']
        # off_acc_mat_weight = (1/np.array(distance_offense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * off_acc_mat)[closest_off]
        # off_vel_mat_weight = (1/np.array(distance_offense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * off_vel_mat)[closest_off]

        def_movement_features = get_player_movement_features(def_df, N)
        # def_acc_mat = def_movement_features['field_weighted_acc']
        def_vel_mat = def_movement_features['field_weighted_velocity'][closest]
        # def_distance_mat = off_movement_features['distance']
        # def_acc_mat_weight = (1/np.array(distance_defense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * def_acc_mat)[closest]
        # def_vel_mat_weight = (1/np.array(distance_defense_from_ballcarrier+0.001)[:, np.newaxis, np.newaxis] * def_vel_mat)[closest]

        ball_movement_features = get_player_movement_features(ball_df, N)
        # ball_acc_mat = ball_movement_features['field_weighted_acc']
        ball_vel_mat = ball_movement_features['field_weighted_velocity']

        off_density = get_player_field_densities(off_df.iloc[closest_off], N)
        def_density = get_player_field_densities(def_df.iloc[closest], N)

        ret = np.stack([
                off_density, def_density, 
                # np.min(off_distance_mat, axis = 0), np.min(def_distance_mat, axis = 0),
                # np.mean(off_distance_mat, axis = 0), np.mean(def_distance_mat, axis = 0),
                np.sum(ball_vel_mat, axis = 0), 
                # np.sum(ball_acc_mat, axis = 0),
                np.sum(off_vel_mat, axis = 0), 
                # np.max(off_acc_mat_weight, axis = 0), np.sum(off_acc_mat_weight, axis = 0), np.std(off_acc_mat, axis = 0),
                np.sum(def_vel_mat, axis = 0)
                # np.max(def_acc_mat_weight, axis = 0), np.sum(def_acc_mat_weight, axis = 0), np.std(def_acc_mat, axis = 0),
                ])
        if not plot:
            return ret
        else:
            fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 10))
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
    
    def predict_tackle_distribution(self, model, without_player_id = 0):
        batch_images = self.get_full_play_tackle_image(N = model.N, without_player_id=without_player_id)
        outputs_all = model.predict_pdf(batch_images)
        return {'estimate' : outputs_all['overall_pred'], 'lower' : outputs_all['lower'], 'upper' : outputs_all['upper'], 
                'all' : outputs_all['all_pred'], 'expected_x_stratified' : outputs_all['expected_stratified'][:, :, 1]}
    
    def get_expected_eop(self, predict_object):
        expected_x_stratified = predict_object['expected_x_stratified'] 
        expected_eop = np.sum(expected_x_stratified, axis = 0)
        return expected_eop
    
    def get_expected_contribution(self, model, original_predict_object, w_omission_all_pred):
        orig_expected_x_stratified = original_predict_object['expected_x_stratified'] # expected value of each model orig
        omit_expected_x_stratified = w_omission_all_pred['expected_x_stratified'] # expected value of each model
        contributions = omit_expected_x_stratified - orig_expected_x_stratified # (num_model, num_frame)
        lower = np.percentile(contributions, q=2.5, axis=0)
        upper = np.percentile(contributions, q=97.5, axis=0)
        estimate = np.mean(contributions, axis = 0)
        return {'lower' : lower, 'upper' : upper, 'estimate' : estimate}
    
    def get_plot_df(self, model):
        # Get df from original (i.e. no player replacement)
        original_no_omit = self.predict_tackle_distribution(model=model, without_player_id = 0)
        exp_eop_orig = self.get_expected_eop(predict_object=original_no_omit)
        outputs_no_omit, lower_no_omit, upper_no_omit = original_no_omit['estimate'], original_no_omit['lower'], original_no_omit['upper']
        exp_contribution, lower_contribution, upper_contribution = np.zeros(len(range(self.min_frame, self.num_frames+1))), np.zeros(len(range(self.min_frame, self.num_frames+1))), np.zeros(len(range(self.min_frame, self.num_frames+1)))
        exp_ret = array_to_field_dataframe(input_array=outputs_no_omit, N=1, marginalized=False)
        exp_ret["type"] = "prob"
        lower_ret = array_to_field_dataframe(input_array=lower_no_omit, N=1, marginalized=False)
        lower_ret['type'] = "lower"
        upper_ret = array_to_field_dataframe(input_array=upper_no_omit, N=1, marginalized=False)
        upper_ret['type'] = "upper"
        ret_tackle_probs = pd.concat([exp_ret, lower_ret, upper_ret], axis = 0)
        ret_tackle_probs['omit'] = 0
        ret_tackle_probs['frameId'] = ret_tackle_probs['frameId']+self.min_frame
        ret_tackle_probs = ret_tackle_probs.pivot(index=['x', 'y', 'frameId', 'omit'], columns='type', values='prob').reset_index()
        ret_contribution = pd.DataFrame({"frameId" : range(self.min_frame, self.num_frames+1), "exp_eop" : exp_eop_orig, 
                                         "exp_contribution" : exp_contribution, "lower_contribution" : lower_contribution, "upper_contribution" : upper_contribution})
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
            exp_eop = self.get_expected_eop(predict_object=original)
            print(f"Expected original EOP frame 1: {exp_eop_orig[0]}")
            print(f"Expected with ommitted EOP frame 1: {exp_eop[0]}")
            original_contribution = self.get_expected_contribution(model=model, original_predict_object=original_no_omit, w_omission_all_pred=original)
            outputs, lower, upper = original['estimate']-outputs_no_omit, original['lower']-lower_no_omit, original['upper']-upper_no_omit
            exp_contribution, lower_contribution, upper_contribution = original_contribution['estimate'], original_contribution['lower'], original_contribution['upper']
            print(f"Expected average contribution: {np.mean(exp_contribution)} 95% CI: [{np.mean(lower_contribution), np.mean(upper_contribution)}]")
            exp_ret = array_to_field_dataframe(input_array=outputs, N=1, marginalized=False)
            exp_ret["type"] = "prob"
            lower_ret = array_to_field_dataframe(input_array=lower, N=1, marginalized=False)
            lower_ret['type'] = "lower"
            upper_ret = array_to_field_dataframe(input_array=upper, N=1, marginalized=False)
            upper_ret['type'] = "upper"
            ret_tackle_probs = pd.concat([exp_ret, lower_ret, upper_ret], axis = 0)
            ret_tackle_probs['omit'] = id
            ret_tackle_probs['frameId'] = ret_tackle_probs['frameId']+self.min_frame
            ret_tackle_probs = ret_tackle_probs.pivot(index=['x', 'y', 'frameId', 'omit'], columns='type', values='prob').reset_index()
            ret_contribution = pd.DataFrame({"frameId" : range(self.min_frame, self.num_frames+1), "exp_eop" : exp_eop, 
                                            "exp_contribution" : exp_contribution, "lower_contribution" : lower_contribution, "upper_contribution" : upper_contribution})
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

def get_player_field_densities(player_df, N, missing_id=0):
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

def get_discrete_pdf_from_mvn(model):
    coordinates = np.floor(model.sample(sample_shape=torch.Size([10000]))).detach().numpy()
    condition_x = (coordinates[:,0] >= 0) & (coordinates[:,0] < 53.3)
    condition_y = (coordinates[:,1] >= 0) & (coordinates[:,1] < 120)
    valid_indices = (condition_x & condition_y)
    coordinates = coordinates[valid_indices, :]
    grid = np.zeros((120, 54), dtype=int)
    x_indices = coordinates[:, 0].astype(int)
    y_indices = coordinates[:, 1].astype(int)
    np.add.at(grid, (y_indices, x_indices), 1)
    discrete_pdf = grid / len(coordinates)
    return discrete_pdf

def get_mixture_pdf(normal_models, pi_models):
    mean = normal_models.mean # (Batch, Nmix, 2)
    cov = normal_models.covariance_matrix # (Batch, Nmix, 2, # (Batch, Nmix, 2))
    probs = pi_models.probs # (Batch, Nmix)
    x, y = torch.meshgrid(torch.linspace(0, 120, 120), torch.linspace(0, 53, 53))
    pdfs = np.zeros((mean.shape[0], 120, 54))
    for batch in range(mean.shape[0]):
        pdf_values = np.array([probs[batch][k].detach().numpy()*get_discrete_pdf_from_mvn(MultivariateNormal(mean[batch, k], cov[batch,k]))
                        for k in range(mean.shape[1])]) # (nmix, num_grid_points)
        pdf = np.sum(pdf_values, axis = 0)
        pdfs[batch,:,:] = pdf
    return pdfs


class TackleNetEnsemble:
    def __init__(self, num_models, N):
        self.models = dict()
        self.N = N
        self.num_models = num_models
        for mod_num in range(num_models):
            with open(f"model_{mod_num}.pkl", 'rb') as f:
                model = pickle.load(f)
            self.models.update({mod_num : model})
    
    def predict_pdf(self, image):
        preds = []
        pis = []
        mus = []
        sigmas = []
        for mod_num in range(self.num_models):
            model = self.models[mod_num]
            pred = model(image)
            # get variables for full ensemble gauccian mixture
            pis.append(pred[0].probs.detach().numpy())
            mus.append(pred[1].mean.detach().numpy())
            sigmas.append(pred[1].covariance_matrix.detach().numpy())
            # predict individually and average
            pred = get_mixture_pdf(pred[1], pred[0])
            preds.append(pred)
        print("-----")
        pis_combined = np.stack(pis)
        pis_new = pis_combined.reshape((pis_combined.shape[1], pis_combined.shape[2]*self.num_models)) # (frames, num_mod*K)
        mus_combined = np.stack(mus)
        mus_new=mus_combined.reshape((mus_combined.shape[1], mus_combined.shape[2]*self.num_models, 2)) # (frames, num_mod*K, 2)
        sigmas_combined = np.stack(sigmas) 
        sigmas_new = sigmas_combined.reshape((sigmas_combined.shape[1], sigmas_combined.shape[2]*self.num_models, 2, 2)) # (frames, num_mod*K, 2, 2)
        
        # calculate expected values x
        expected_vec_stratified = np.tile(pis_combined[..., np.newaxis], 2)*mus_combined # (num_mod, frames, K, 2)
        expected_stratified = np.sum(expected_vec_stratified, axis = 2) # (num_mod, frames, 2) 

        # create object
        mixture_ensemble_object = (OneHotCategorical(probs=torch.Tensor(pis_new/self.num_models)), 
                                   MultivariateNormal(loc=torch.Tensor(mus_new), covariance_matrix=torch.Tensor(sigmas_new)))
        # Get preds and confidence
        preds = np.stack(preds)
        lower = np.percentile(preds, q = 2.75, axis = 0) 
        overall_pred = np.mean(preds, axis = 0)
        upper = np.percentile(preds, q = 97.5, axis = 0)
        return {'overall_pred' : overall_pred, 'lower' : lower, 'upper' : upper, 
                'all_pred' : preds, 'mixture_return' : mixture_ensemble_object, 'expected_stratified' : expected_stratified}

    

class GaussianMixtureEnsembleLogLoss(nn.Module):
    def __init__(self):
        super(GaussianMixtureLoss, self).__init__()

    def forward(self, output, y):
        pi_model, normal_model = output[0], output[1]
        loglik = normal_model.log_prob(y.unsqueeze(1).expand_as(normal_model.loc))
        loss = -torch.logsumexp(torch.log(pi_model.probs) + loglik, dim=1)
        return loss.mean()

class GaussianMixtureLoss(nn.Module):
    def __init__(self):
        super(GaussianMixtureLoss, self).__init__()

    def forward(self, output, y):
        pi_model, normal_model = output[0], output[1]
        loglik = normal_model.log_prob(y.unsqueeze(1).expand_as(normal_model.loc))
        loss = -torch.logsumexp(torch.log(pi_model.probs) + loglik, dim=1)
        return loss.mean()

                
class BivariateGaussianMixture(nn.Module):
    def __init__(self, nmix, full_cov=True):
        super().__init__()
        self.nmix = nmix
        self.conv1 = nn.Conv2d(5, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride=1) 
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride=1)  
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride=1)          
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.8)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.mu_net = nn.Sequential(
                            nn.Linear(5800, 100),
                            nn.ReLU(),
                            nn.Linear(100, 50),
                            nn.ReLU(),
                            nn.Linear(50, nmix * 2),
                        )
        self.cov_net = nn.Sequential(
                            nn.Linear(5800, 100),
                            nn.ReLU(),
                            nn.Linear(100, 50),
                            nn.ReLU(),
                            nn.Linear(50, nmix * 2),
                        )
        self.pi_net = nn.Sequential(
                            nn.Linear(5800, 100),
                            nn.ReLU(),
                            nn.Linear(100, 50),
                            nn.ReLU(),
                            nn.Linear(50, nmix),
                        )
        self.elu = nn.ELU()
        init.constant_(self.mu_net[-1].bias, 35)
        init.constant_(self.cov_net[-1].bias, 10)


    def forward(self, x):
        # print(f"OG Shape: {x.shape}")
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        
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

    




        






