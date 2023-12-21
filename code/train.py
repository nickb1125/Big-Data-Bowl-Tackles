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
from objects import plot_field, play, TackleAttemptDataset, plot_predictions, TackleNetEnsemble, BivariateGaussianMixture, GaussianMixtureLoss
import pickle
from torchvision import transforms, utils, models
from torch.optim import lr_scheduler


# Define Loss
# criterion = nn.BCELoss()
train = False
nmix=5
criterion = GaussianMixtureLoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# open test set dict:
with open("data/test_frame_dict.pkl", 'rb') as f:
    test_frame_dict = pickle.load(f)
from_frame_end_values = list(test_frame_dict.keys())
from_frame_end_values.sort()

if train:
    # Train bagged models
    bag = 0
    while bag < 10:
        with open(f'data/tackle_image_bag_{bag}.pkl', 'rb') as f:
            tackle_dataset = pickle.load(f)

        train_data, val_data = torch.utils.data.random_split(tackle_dataset, [0.9, 0.1])
        print(len(train_data))
        print(len(val_data))

        # Create Data Loader
        train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64)

        # Create Model
        model = BivariateGaussianMixture(nmix=nmix, full_cov=True)
        # model = TackleNet(N = 5, nvar = 5)
        # model = models.resnet18(weights=models.ResNet50_Weights.DEFAULT).to(device)
        # fc = OnlyFC(N=5)
        # model.fc = fc
        # model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Define the optimizer (e.g., Stochastic Gradient Descent)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        # Training loop
        num_epochs = 25

        print(f"Training TackleNet bag {bag}...")
        print("---------------------")
        losses = []
        val_losses = []
        val_accuracy = []
        start_over = False
        for epoch in range(num_epochs):
            if epoch >= 5:
                for param in model.resnet.parameters():
                    model.resnet.requires_grad = False
            else:
                for param in model.resnet.parameters():
                    model.resnet.requires_grad = True
            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                try:
                    output = model(X_batch)
                except:
                    print("Gradient Exploded, trying to train this model again...")
                    start_over = True
                    break
                loss = criterion(output, y_batch)
                # print(loss)
                loss.backward()
                optimizer.step()
            if start_over:
                break
            # After training, you can use the model to make predictions
            losses_1 = []
            with torch.no_grad():  # Disable gradient computation for inference
                for X_batch, y_batch in train_dataloader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    losses_1.append(loss)
                loss = sum(losses_1)
                losses.append(loss)
                print(f"Epoch: {epoch}")
                print(f"Train Loss: {loss}")
            val_losses = []
            with torch.no_grad():  # Disable gradient computation for inference
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    val_loss = criterion(outputs, y_batch)
                    val_losses.append(val_loss)
                val_loss = sum(val_losses)
            print(f"Validation Loss: {val_loss}")
            scheduler.step()
        if start_over:
            continue

        ### For test set
        test_losses = []
        with torch.no_grad():  # Disable gradient computation for inference
            for from_end_frame in from_frame_end_values:
                with open(f'data/test_tackle_images_{from_end_frame}_from_end.pkl', 'rb') as f:
                    test_dataset = pickle.load(f)
                test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
                predictions = []
                counter = 0
                with torch.no_grad():  # Disable gradient computation for inference
                    for X_batch, y_batch in test_dataloader:
                        counter += X_batch.shape[0]
                        outputs = model(X_batch)
                        test_loss = criterion(outputs, y_batch)
                        test_losses.append(test_loss)
                test_loss = sum(test_losses)
                print("---------------------------------")
                print(f"Number of test samples for {from_end_frame} frames from EOP: {counter}")
                print(f"Test Loss for {from_end_frame} frames from EOP: {test_loss}")
        bag += 1


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
ensemble_model = TackleNetEnsemble(num_models=10, N=1)
i = 0
print(f"ENSEMBLE MODEL Final Test Preformance:")
print("---------------------------------")
test_losses = []
for from_end_frame in from_frame_end_values:
    with open(f'data/test_tackle_images_{from_end_frame}_from_end.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    counter = 0
    with torch.no_grad():  # Disable gradient computation for inference
        for X_batch, y_batch in test_dataloader:
            counter += X_batch.shape[0]
            pred = ensemble_model.predict_pdf(X_batch)
            output_model = pred['mixture_return']
            outputs = pred['overall_pred']
            test_loss = criterion(output_model, y_batch)
            test_losses.append(test_loss)
            if (i == 0):
                plot_field(outputs, y_batch)
            i += 1
    test_loss = sum(test_losses)
    print("---------------------------------")
    print(f"Number of test samples for {from_end_frame} frames from EOP: {counter}")
    print(f"Test Loss for {from_end_frame} frames from EOP: {test_loss}")
