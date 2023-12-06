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
from objects import play, TackleAttemptDataset, TackleNet, plot_predictions, imageCache
import pickle

print("Cacheing all play training data (imcluding with player ommissions)")
print("-----------------")

image_cache = imageCache(N = 5)
image_cache.populate()

with open(f"data/image_cache_5.pkl", f'wb') as outp:  # Overwrites any existing file.
    pickle.dump(image_cache, outp, pickle.HIGHEST_PROTOCOL)

