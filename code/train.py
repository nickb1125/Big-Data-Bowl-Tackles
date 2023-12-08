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

with open(f'data/tackle_images_5_output_5_test.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define Loss
criterion = nn.BCELoss()

# Train bagged models
for bag in range(100):
    with open(f'data/tackle_images_5_output_5_bag_{bag}_stratified.pkl', 'rb') as f:
        tackle_dataset = pickle.load(f)

    train_data, val_data = torch.utils.data.random_split(tackle_dataset, [0.9, 0.1])
    print(len(train_data))
    print(len(val_data))

    # Create Data Loader
    train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    # Create Model
    model = TackleNet(N = 5, nvar = 12)

    # Define the optimizer (e.g., Stochastic Gradient Descent)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    num_epochs = 10

    print(f"Training TackleNet bag {bag}...")
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
        exact_correct = []
        with torch.no_grad():  # Disable gradient computation for inference
            i = 0
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                predictions.append(outputs.flatten())
                true_labels.append(y_batch.flatten())
                # for accuracy:
                max_values, _ = torch.max(outputs, dim=-1, keepdim=True)
                max_values, _ = torch.max(max_values, dim=-2, keepdim=True)
                masks = torch.where(outputs == max_values, 1, 0)
                equivalence_values = np.all(masks.detach().numpy() == y_batch.detach().numpy(), axis=(1, 2)).tolist()
                exact_correct.extend(equivalence_values)
                i += 1
            predictions = torch.cat(predictions, dim=0) 
            true_labels = torch.cat(true_labels, dim=0) 
            val_loss = criterion(predictions, true_labels)
            val_losses.append(val_loss.detach())
        acc = sum(exact_correct)/len(exact_correct)
        val_accuracy.append(acc)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {acc}")

    ### For test set
    predictions = []
    true_labels = []
    exact_correct = []
    with torch.no_grad():  # Disable gradient computation for inference
        i = 0
        for X_batch, y_batch in test_dataloader:
            outputs = model(X_batch)
            predictions.append(outputs.flatten())
            true_labels.append(y_batch.flatten())
            # for accuracy:
            max_values, _ = torch.max(outputs, dim=-1, keepdim=True)
            max_values, _ = torch.max(max_values, dim=-2, keepdim=True)
            masks = torch.where(outputs == max_values, 1, 0)
            equivalence_values = np.all(masks.detach().numpy() == y_batch.detach().numpy(), axis=(1, 2)).tolist()
            exact_correct.extend(equivalence_values)
            i += 1
        predictions = torch.cat(predictions, dim=0) 
        true_labels = torch.cat(true_labels, dim=0) 
        test_loss = criterion(predictions, true_labels)
    test_acc = sum(exact_correct)/len(exact_correct)
    print(f"Bag {bag} Final Test Preformance:")
    print("---------------------------------")
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")


    # plt.figure(figsize=(10, 5))
    # plt.plot(losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(val_accuracy, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Accuracy')
    # plt.title('Validation Binary Prediction Accuracy Over Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    with open(f"model_{bag}.pkl", f'wb') as outp:  # Overwrites any existing file.
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

### Final ensemble on test
ensemble_model = TackleNetEnsemble(num_models=10, N = 5)
predictions = []
true_labels = []
exact_correct = []
with torch.no_grad():  # Disable gradient computation for inference
    i = 0
    for X_batch, y_batch in test_dataloader:
        outputs = ensemble_model.predict_pdf(X_batch)
        if (i == 0):
            plot_predictions(outputs, y_batch)
        predictions.append(outputs.flatten())
        true_labels.append(y_batch.flatten())
        # for accuracy:
        max_values, _ = torch.max(outputs, dim=-1, keepdim=True)
        max_values, _ = torch.max(max_values, dim=-2, keepdim=True)
        masks = torch.where(outputs == max_values, 1, 0)
        equivalence_values = np.all(masks.detach().numpy() == y_batch.detach().numpy(), axis=(1, 2)).tolist()
        exact_correct.extend(equivalence_values)
        i += 1
    predictions = torch.cat(predictions, dim=0) 
    true_labels = torch.cat(true_labels, dim=0) 
    test_loss = criterion(predictions, true_labels)
test_acc = sum(exact_correct)/len(exact_correct)
print(f"ENSEMBLE MODEL Final Test Preformance:")
print("---------------------------------")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

