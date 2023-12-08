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
from objects import play, TackleAttemptDataset, TackleNet, plot_predictions, TackleNetEnsemble, update
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

game_id = 2022100212
play_id = 2007

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

model = TackleNetEnsemble(num_models = 10, N = 5)

all_pred = play_object.get_contribution_matricies(model = model, to_df = True, marginal_x=False)
all_pred.to_csv(f"{game_id}_{play_id}.csv")



### plot 3d predictions with prediction interval

# Set up the initial figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

# Dummy scatter plot to generate legend
surf = ax.scatter([], [], [], c=[], cmap="Reds", s=300, edgecolors="black", linewidth=0.5, marker="o", label='Legend Label')

# Set label='' to prevent automatic legend creation
ax.legend()

# Set the specific frames you want to animate (frames 25 through 30)
frames_to_animate = pred_df.frameId.unique()

# Create the animation
animation = FuncAnimation(fig, update, frames=frames_to_animate, fargs=(pred_df, surf))

# Save the animation as a GIF
animation.save('animation.gif', writer='pillow', fps=10)