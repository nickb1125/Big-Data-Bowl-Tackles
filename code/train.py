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
from objects import euclidean_distance, play, TackleAttemptDataset, TackleNet, plot_predictions, play_cache
import pickle

with open('data/tackle_images.pkl', 'rb') as f:
    tackle_dataset = pickle.load(f)

train_data, val_data = torch.utils.data.random_split(tackle_dataset, [0.9, 0.1])

# Create Data Loader
train_dataloader = DataLoader(train_data, batch_size=250, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

# Define the loss function (you can choose an appropriate loss function for your task)
criterion = nn.BCELoss()

# Create Model
model = TackleNet(N = 10, nvar = 24)

# Define the optimizer (e.g., Stochastic Gradient Descent)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 25

print("Training TackleNet...")
print("---------------------")
losses = []
val_losses = []
val_accuracy = []
for epoch in range(num_epochs):
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.flatten(), y_batch.flatten())
        loss.backward()
        optimizer.step()

    # After training, you can use the model to make predictions
    predictions = []
    true_labels = []
    with torch.no_grad():  # Disable gradient computation for inference
        for X_batch, y_batch in train_dataloader:
            outputs = model(X_batch)
            predictions.append(outputs.flatten())
            true_labels.append(y_batch.flatten())
        predictions = torch.cat(predictions, dim=0) 
        true_labels = torch.cat(true_labels, dim=0) 
        loss = criterion(predictions, true_labels)
        losses.append(loss.detach())
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {loss}")
    predictions = []
    true_labels = []
    prediction_correct = []
    with torch.no_grad():  # Disable gradient computation for inference
        i = 0
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            if (i == 0) & (epoch == num_epochs-1):
                plot_predictions(outputs, y_batch)
            predictions.append(outputs.flatten())
            true_labels.append(y_batch.flatten())
            i += 1
        predictions = torch.cat(predictions, dim=0) 
        true_labels = torch.cat(true_labels, dim=0) 
        val_loss = criterion(predictions, true_labels)
        val_losses.append(val_loss.detach())
        max_values, _ = torch.max(outputs, dim=-1, keepdim=True)
        max_values, _ = torch.max(max_values, dim=-2, keepdim=True)
        masks = torch.where(outputs == max_values, 1, 0)
        equivalence_tensor = torch.all(y_batch == masks, dim=0)
        equivalence_values = torch.all(equivalence_tensor, dim=1).numpy().tolist()
        max_values, _ = torch.max(outputs, dim=-1, keepdim=True)
        max_values, _ = torch.max(max_values, dim=-2, keepdim=True)
        masks = torch.where(outputs == max_values, 1, 0)
        equivalence_values = np.all(masks.detach().numpy() == y_batch.detach().numpy(), axis=(1, 2)).tolist()
        prediction_correct.extend(equivalence_values)
    acc = sum(prediction_correct)/len(prediction_correct)
    val_accuracy.append(acc)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {acc}")

with open("model.pkl", f'wb') as outp:  # Overwrites any existing file.
    pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)