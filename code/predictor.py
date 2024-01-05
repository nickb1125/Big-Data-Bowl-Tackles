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
from objects import play, TackleAttemptDataset, TackleNetEnsemble, array_to_field_dataframe
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap


prepredicted = False
main_plot = True
contribution_plot = False
game_id=2022091108
play_id=1948

print("Loading base data")
print("-----------------")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])

colors = [(0, 0, 0), (1, 0, 0)]  # Grey to Red
custom_cmap = LinearSegmentedColormap.from_list("CustomReds", colors, N=256)
colors = [(0, 0, 1), (0.5, 0.5, 0.5), (1, 0, 0)]  # Blue to Grey to Red
custom_cmap_two_way = LinearSegmentedColormap.from_list('blue_to_grey_to_red', colors, N=256)
x_range = np.linspace(0, 10, 10)  # 100 points between 0 and 10
y_range = np.linspace(0, 54, 10)  # 100 points between 0 and 54
x_td1, y_td1 = np.meshgrid(x_range, y_range)
x_range = np.linspace(110, 120, 10)
x_td2, y_td2 = np.meshgrid(x_range, y_range)
x_range = np.linspace(0, 120, 10)
x_field, y_field = np.meshgrid(x_range, y_range)

if not prepredicted:
    print("Predicting")
    print("-----------------")

    # Get all defensive players to omit over play

    play_object = play(game_id, play_id, tracking)
    def_df = play_object.refine_tracking(frame_id = play_object.min_frame)["Defense"]
    def_ids = def_df.nflId.unique()

    # Define model

    model = TackleNetEnsemble(num_models = 10, N = 3, nmix=5)

    all_pred = play_object.get_plot_df(model = model)
    all_pred.to_csv(f"{game_id}_{play_id}.csv")
    feat = play_object.get_full_play_tackle_image(N = 1)
    feat_df = array_to_field_dataframe(input_array=feat, N=1, for_features=True)
    feat_df.to_csv(f"/Users/nickbachelder/Desktop/Personal Code/Kaggle/Tackles/features_{game_id}_{play_id}.csv")
    closest_three_def = play_object.get_closest_three_def_from_eop()
else:
    print("Reading in data.")
    all_pred = pd.read_csv(f"{game_id}_{play_id}.csv")
    play_object = play(game_id, play_id, tracking)
    closest_three_def = play_object.get_closest_three_def_from_eop()
    print("Done.")


### plot 3d predictions with prediction interval
def update_main(frame_id, dataframe, scatter, ax):
    tracking_now = tracking.query("gameId == @game_id & playId == @play_id & frameId == @frame_id")[['x', 'y', 'type']]
    tracking_now['x'] = 120 - tracking_now['x']
    tracking_now_off = tracking_now.query("type == 'Offense'")
    tracking_now_def = tracking_now.query("type == 'Defense'")
    tracking_now_ball = tracking_now.query("type == 'Ball'")
    
    dataframe_now = dataframe.query("(frameId == @frame_id) & (omit == 0)")
    fx = sorted(dataframe_now['x'].unique())
    fy = sorted(dataframe_now['y'].unique())
    z, zerror_lower, zerror_upper = [], [], []

    overall_max_z = np.max(dataframe.query("(omit == 0)").prob.values)*2

    for y_val in fy:  # Reverse the order of y values
        row_data, error_lower_data, error_upper_data = [], [], []

        for x_val in fx:  # Reverse the order of x values
            subset = dataframe_now[(dataframe_now['x'] == x_val) & (dataframe_now['y'] == y_val)]
            row_data.append(subset['prob'].values[0])
            error_lower_data.append(subset['lower'].values[0])
            error_upper_data.append(subset['upper'].values[0])
        z.append(row_data)
        zerror_lower.append(error_lower_data)
        zerror_upper.append(error_upper_data)
    
    z = np.fliplr(z)
    zerror_lower = np.fliplr(zerror_lower)
    zerror_upper = np.fliplr(zerror_upper)

    x, y = np.meshgrid(fx, fy)  # Reverse the order of x and y values
    ax.cla()
    ax.plot_surface(x, y, np.array(z), cmap=custom_cmap, alpha=1)
    ax.plot_surface(x_field, y_field, np.full_like(np.zeros((10,10)), overall_max_z), color="green", alpha=0.5)
    ax.plot_surface(x_td1, y_td1, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.8)
    ax.plot_surface(x_td2, y_td2, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.8)

    # tick marks
    y_range = np.linspace(0, 54, 10) 
    for yardline in range(15, 110, 5):
        x_range = np.linspace(yardline-0.1, yardline+0.1, 10)
        x_yard, y_yard = np.meshgrid(x_range, y_range)
        ax.plot_surface(x_yard, y_yard, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.5)
    tick_1_y = (54/2) + (6.16/2)
    tick_2_y = (54/2) - (6.16/2)
    for tick in range(10, 110, 1):
        x_range = np.linspace(tick-0.1, tick+0.1, 10)
        for value in [tick_1_y, tick_2_y]:
            y_range = np.linspace(value-0.5, value+0.5, 10)
            x_yard, y_yard = np.meshgrid(x_range, y_range)
            ax.plot_surface(x_yard, y_yard, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.5)

    ax.scatter(tracking_now_off['x'], tracking_now_off['y'], overall_max_z, depthshade=False, c='grey', marker='D', label='Tracking Points', s = 150, alpha = 1)
    ax.scatter(tracking_now_def['x'], tracking_now_def['y'], overall_max_z, depthshade=False, c='dodgerblue', marker='D', label='Tracking Points', s = 150, alpha = 1)
    ax.scatter(tracking_now_ball['x'], tracking_now_ball['y'], overall_max_z, depthshade=False, c='red', marker='D', label='Tracking Points', s = 200, alpha = 1)

    ax.set_xlabel('Yardline')
    ax.set_title(f'Tackle Probability Density with Prediction Interval', fontweight='bold', fontsize = 40)

    for i in range(len(fy)):
        ax.plot_surface(x[i], y[i], np.array([zerror_upper[i], zerror_lower[i]]), color='white', alpha=0.05)

    ax.set_box_aspect([2, 1, 0.5])
    ax.grid(False)
    ax.set_zlim([0, overall_max_z])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.w_xaxis.line.set_lw(0)
    ax.w_yaxis.line.set_lw(0)
    ax.w_zaxis.line.set_lw(0)

    # Return a sequence of artists to be drawn
    return scatter,

def update(frame_id, dataframe, axs, omit_values):
    tracking_now = tracking.query("gameId == @game_id & playId == @play_id & frameId == @frame_id")[['displayName', 'nflId', 'x', 'y', 'type']]
    tracking_now['x'] = 120 - tracking_now['x']
    tracking_now_def = tracking_now.query("type == 'Defense'")
    tracking_now_ball = tracking_now.query("type == 'Ball'")
    overall_max_z = np.max(dataframe.query("(omit != 0)").prob.values)

    for ax, omit in zip(axs.flatten(), omit_values):
        ax.clear()

        dataframe_now = dataframe.query("(frameId == @frame_id) & (omit == @omit)")
        dataframe_now_metrics = dataframe_now[['frameId', 'exp_contribution']].drop_duplicates().exp_contribution.values
        min_z = np.min(dataframe.query("(omit == @omit)").prob)
        max_z = np.max(dataframe.query("(omit == @omit)").prob)
        this_def_player = tracking_now_def.query("nflId == @omit")
        player_name = this_def_player.displayName.values[0]
        
        fx = sorted(dataframe_now['x'].unique())
        fy = sorted(dataframe_now['y'].unique())
        z, zerror_lower, zerror_upper = [], [], []

        for y_val in fy:
            row_data, error_lower_data, error_upper_data = [], [], []

            for x_val in fx:
                subset = dataframe_now[(dataframe_now['x'] == x_val) & (dataframe_now['y'] == y_val)]
                row_data.append(subset['prob'].values[0])
                error_lower_data.append(subset['lower'].values[0])
                error_upper_data.append(subset['upper'].values[0])

            z.append(row_data)
            zerror_lower.append(error_lower_data)
            zerror_upper.append(error_upper_data)
        
        z = np.fliplr(z)
        zerror_lower = np.fliplr(zerror_lower)
        zerror_upper = np.fliplr(zerror_upper)

        x, y = np.meshgrid(fx, fy)
        ax.plot_surface(x, y, np.array(z), cmap=custom_cmap_two_way, alpha=0.5, label=f'Omit={omit}', norm = Normalize(vmin=-max_z, vmax=max_z))

        # tick marks
        y_range = np.linspace(0, 54, 10) 
        for yardline in range(15, 110, 5):
            x_range = np.linspace(yardline-0.1, yardline+0.1, 10)
            x_yard, y_yard = np.meshgrid(x_range, y_range)
            ax.plot_surface(x_yard, y_yard, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.5)
        tick_1_y = (54/2) + (6.16/2)
        tick_2_y = (54/2) - (6.16/2)
        for tick in range(10, 110, 1):
            x_range = np.linspace(tick-0.1, tick+0.1, 10)
            for value in [tick_1_y, tick_2_y]:
                y_range = np.linspace(value-0.5, value+0.5, 10)
                x_yard, y_yard = np.meshgrid(x_range, y_range)
                ax.plot_surface(x_yard, y_yard, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.5)
    
        ax.scatter(this_def_player['x'], this_def_player['y'], overall_max_z+overall_max_z/10000, c = dataframe_now_metrics, cmap=custom_cmap_two_way, marker='o', 
                   label='Defense', alpha=1, s = 300, norm = Normalize(vmin=-5, vmax=5))
        ax.scatter(tracking_now_ball['x'], tracking_now_ball['y'], overall_max_z+overall_max_z/10000, c='red', marker='o', label='Ball Carrier', alpha=1,s=300)
        ax.plot_surface(x_field, y_field, np.full_like(np.zeros((10,10)), overall_max_z), color="green", alpha=0.5)
        ax.plot_surface(x_td1, y_td1, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.8)
        ax.plot_surface(x_td2, y_td2, np.full_like(np.zeros((10,10)), overall_max_z), color="black", alpha=0.8)

        ax.set_title(f'{player_name}', fontsize=50, pad=20, loc='center', y=0.1)

        for i in range(len(fy)):
            ax.plot_surface(x[i], y[i], np.array([zerror_upper[i], zerror_lower[i]]), color='grey', alpha=0.01)

        ax.set_box_aspect([2, 1, 1])
        ax.grid(False)
        ax.set_zlim([-overall_max_z, overall_max_z])

        # Hide x, y, and z axis ticks and lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.w_xaxis.line.set_lw(0)
        ax.w_yaxis.line.set_lw(0)
        ax.w_zaxis.line.set_lw(0)

    return axs


if contribution_plot:
    # Set the values for omit
    play_object = play(game_id, play_id, tracking)
    def_df = play_object.refine_tracking(frame_id = play_object.min_frame)["Defense"]
    omit_values = closest_three_def
    print(f"Completing plots for defenders: {closest_three_def}")
    fig, axs = plt.subplots(1, 3,  figsize=(50, 20), subplot_kw={'projection': '3d', 'computed_zorder' : False}, constrained_layout=True)
    axs = axs.flatten()
    fig.suptitle('Defensive Player Contributions to Spatial Tackle Density', fontsize=40, y= 1, fontweight='bold')
    # Dummy scatter plot to generate legend
    surf = axs[0].scatter([], [], [], c=[], cmap=custom_cmap_two_way, s=300, edgecolors="black", linewidth=0.5, marker="o", label='Legend Label')

    # Set label='' to prevent automatic legend creation
    axs[0].legend()

    # Set the specific frames you want to animate 
    frames_to_animate = all_pred.frameId.unique()

    # Create the animation
    print("Animating Contributions.")
    animation = FuncAnimation(fig, update, frames=frames_to_animate, fargs=(all_pred, axs, omit_values), blit=False)
    print("Done.")

    # Save the animation as a GIF
    print("Saving.")
    animation.save(f'{game_id}_{play_id}_contribution_animation.gif', writer='pillow', fps=10)
    print("Done.")

if main_plot:
    # Set up the initial figure
    fig = plt.figure(figsize=(30, 20), constrained_layout=True)
    ax = fig.add_subplot(projection='3d', computed_zorder=False)

    # Dummy scatter plot to generate legend
    surf = ax.scatter([], [], [], c=[], cmap=custom_cmap, s=300, edgecolors="black", linewidth=0.5, marker="o", label='Legend Label')

    # Set label='' to prevent automatic legend creation
    ax.legend()

    # Set the specific frames you want to animate (frames 25 through 30)
    frames_to_animate = all_pred.frameId.unique()

    # Create the animation
    print("Animating.")
    animation = FuncAnimation(fig, update_main, frames=frames_to_animate, fargs=(all_pred, surf, ax), blit=False)
    print("Done.")

    # Save the animation as a GIF
    print("Saving.")
    animation.save(f'{game_id}_{play_id}_animation.gif', writer='pillow', fps=10)
    print("Done.")







