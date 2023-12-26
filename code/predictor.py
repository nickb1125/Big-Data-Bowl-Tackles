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
game_id = 2022090800
play_id = 101

if not prepredicted:
    print("Loading base data")
    print("-----------------")

    # Read and combine tracking data for all weeks
    tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])


    print("Predicting")
    print("-----------------")

    # Get all defensive players to omit over play

    play_object = play(game_id, play_id, tracking)
    def_df = play_object.refine_tracking(frame_id = play_object.min_frame)["Defense"]
    def_ids = def_df.nflId.unique()

    # Define model

    model = TackleNetEnsemble(num_models = 10, N = 1, nmix=5)

    all_pred = play_object.get_plot_df(model = model)
    all_pred.to_csv(f"{game_id}_{play_id}.csv")
    feat = play_object.get_full_play_tackle_image(N = 1)
    feat_df = array_to_field_dataframe(input_array=feat, N=1, for_features=True)
    feat_df.to_csv(f"features_{game_id}_{play_id}.csv")
else:
    print("Reading in data.")
    all_pred = pd.read_csv(f"{game_id}_{play_id}.csv")
    print("Done.")


### plot 3d predictions with prediction interval

def update(frame_id, dataframe, scatter):
    center_density = dataframe.loc[dataframe.prob == max(dataframe.prob)][['x', 'y']]
    center_x, center_y = center_density.x.reset_index(drop=1)[0], center_density.y.reset_index(drop=1)[0]
    dataframe_now = dataframe
    fx = sorted(dataframe_now['x'].unique())[2::1]
    fy = sorted(dataframe_now['y'].unique())[4::1]
    z, zerror_lower, zerror_upper = [], [], []
    dataframe_now = dataframe.query("(frameId == @frame_id)")

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

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('End of Play Probability')
    ax.set_title('Tackle Probability with Prediction Interval')

    for i in range(len(fy)):
        ax.plot_surface(x[i], y[i], np.array([zerror_upper[i], zerror_lower[i]]), color='grey', alpha=0.01)

    ax.set_box_aspect([2, 1, 1])
    # ax.grid(False)
    ax.set_zlim([0, 0.007])

    return scatter

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
animation = FuncAnimation(fig, update, frames=frames_to_animate, fargs=(all_pred, surf))
print("Done.")


# Save the animation as a GIF
print("Saving.")
animation.save('animation.gif', writer='pillow', fps=10)
print("Done.")