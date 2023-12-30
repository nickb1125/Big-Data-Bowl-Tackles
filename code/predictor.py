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
from objects import play, TackleAttemptDataset, plot_predictions, TackleNetEnsemble, array_to_field_dataframe
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.colors import Normalize

prepredicted = False
game_id = 2022103012
play_id = 1919

print("Loading base data")
print("-----------------")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])

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
else:
    print("Reading in data.")
    all_pred = pd.read_csv(f"{game_id}_{play_id}.csv")
    print("Done.")


### plot 3d predictions with prediction interval
def update(frame_id, dataframe, scatter, ax):
    tracking_now = tracking.query("gameId == @game_id & playId == @play_id & frameId == @frame_id")[['x', 'y', 'type']]
    tracking_now_off = tracking_now.query("type == 'Offense'")
    tracking_now_def = tracking_now.query("type == 'Defense'")
    tracking_now_ball = tracking_now.query("type == 'Ball'")
    
    dataframe_now = dataframe.query("(frameId == @frame_id) & (omit == 0)")
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

    x, y = np.meshgrid(fx, fy)
    ax.cla()
    ax.plot_surface(x, y, np.array(z), cmap="Reds", alpha=1)
    ax.scatter(tracking_now_off['x'], tracking_now_off['y'], np.max(z), c='black', marker='o', label='Tracking Points')
    ax.scatter(tracking_now_def['x'], tracking_now_def['y'], np.max(z), c='grey', marker='o', label='Tracking Points')
    ax.scatter(tracking_now_ball['x'], tracking_now_ball['y'], np.max(z), c='red', marker='o', label='Tracking Points')
    ax.plot_surface(x, y, np.full_like(z, np.max(z)), color="grey", alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('End of Play Probability')
    ax.set_title('Tackle Probability with Prediction Interval')

    for i in range(len(fy)):
        ax.plot_surface(x[i], y[i], np.array([zerror_upper[i], zerror_lower[i]]), color='grey', alpha=0.01)

    ax.set_box_aspect([2, 1, 1])
    ax.grid(False)
    ax.set_zlim([0, np.max(z)])

    # Return a sequence of artists to be drawn
    return scatter,

# Set up the initial figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# Dummy scatter plot to generate legend
surf = ax.scatter([], [], [], c=[], cmap="Reds", s=300, edgecolors="black", linewidth=0.5, marker="o", label='Legend Label')

# Set label='' to prevent automatic legend creation
ax.legend()

# Set the specific frames you want to animate (frames 25 through 30)
frames_to_animate = all_pred.frameId.unique()

# Create the animation
print("Animating.")
animation = FuncAnimation(fig, update, frames=frames_to_animate, fargs=(all_pred, surf, ax), blit=False)
print("Done.")

# Save the animation as a GIF
print("Saving.")
animation.save('animation.gif', writer='pillow', fps=10)
print("Done.")