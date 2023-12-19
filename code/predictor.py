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
from objects import play, TackleAttemptDataset, TackleNet, plot_predictions, TackleNetEnsemble
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

game_id = 2022110605
play_id = 2397

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

model = TackleNetEnsemble(nvar = 10, N = 5)

all_pred = play_object.get_plot_df(model = model)
all_pred.to_csv(f"{game_id}_{play_id}.csv")



### plot 3d predictions with prediction interval

def update(frame_id, dataframe, surf):
    ax.clear()

    center_density = dataframe.loc[dataframe.prob == max(dataframe.prob)][['x', 'y']]
    center_x, center_y = center_density.x.reset_index(drop=1)[0], center_density.y.reset_index(drop=1)[0]
    dataframe_now = dataframe.query(
        "(x > @center_x - 40) & (x < @center_x + 40) & (y > @center_y - 20) & (y < @center_y + 20) & (frameId == @frame_id)")

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

    surf = ax.scatter3D(x, y, z, c=z, cmap="Reds", s=300, edgecolors="black", linewidth=0.5, marker="o", label='')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('End of Play Probability')
    ax.set_title('Tackle Probability with Prediction Interval')
    ax.view_init(30, 300)
    ax.set_zlim([0, 1])

    for i in range(len(fy)):
        for xval, yval, zval, zerr_lower, zerr_upper in zip(x[i], y[i], z[i], zerror_lower[i], zerror_upper[i]):
            ax.plot([xval, xval], [yval, yval], [zerr_upper, zerr_lower], marker="_", color='k', label = '')
    ax.set_box_aspect([2, 1, 1])

    return surf

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
animation = FuncAnimation(fig, update, frames=frames_to_animate, fargs=(all_pred, surf))

# Save the animation as a GIF
animation.save('animation.gif', writer='pillow', fps=10)