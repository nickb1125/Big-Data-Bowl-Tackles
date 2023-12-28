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
from sklearn.model_selection import KFold
import statistics

# Make record keeper
records = []

# Define Loss
# criterion = nn.BCELoss()
validate_each_epoch_cv = True
plot = False
train = True
cross_validate = False
criterion = GaussianMixtureLoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if (not train) or (not cross_validate):
    nmix = 5

# open test set dict:
with open("data/test_frame_dict.pkl", 'rb') as f:
    test_frame_dict = pickle.load(f)
from_frame_end_values = list(test_frame_dict.keys())
from_frame_end_values.sort()

if train:
    if cross_validate:
        # Cross validate one model to find nmix
        print("Cross Validating to select nmix...")
        print("---------------------")
        num_splits = 3
        with open(f'data/tackle_image_bag_1.pkl', 'rb') as f:
            tackle_dataset = pickle.load(f)
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
        nmix_scores = []
        nmixes = [5, 10, 15]
        for nmix in nmixes:
            print(f"Cross Validating for nmix = {nmix}")
            print("------------------")
            # Perform K-fold cross-validation
            overall_val_losses = []
            for fold, (train_index, val_index) in enumerate(kf.split(tackle_dataset)):
                print(f"Completing Fold {fold}")
                print("---------------------")
                train_data, val_data = [tackle_dataset[i] for i in train_index.tolist()], [tackle_dataset[i] for i in val_index.tolist()]
                print(f"Cross Validating Train Size For Fold {fold}: {len(train_data)}")
                print(f"Cross Validating Validation Size For Fold {fold}: {len(val_data)}")
                train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=64)
                model = BivariateGaussianMixture(nmix=nmix)
                optimizer = optim.Adam(model.parameters(), lr=0.0005)
                num_epochs = 20
                losses = []
                start_over = False
                j = 0
                for epoch in range(num_epochs):
                    j += 1
                    counter = 0
                    train_losses = []
                    for X_batch, y_batch in train_dataloader:
                        counter += X_batch.shape[0]
                        optimizer.zero_grad()
                        try:
                            output = model(X_batch)
                        except:
                            print("Gradient Exploded, trying to train this model again...")
                            start_over = True
                            break
                        loss = criterion(output, y_batch)
                        train_losses.append(loss.detach().item())
                        loss.backward()
                        optimizer.step()
                    if start_over:
                        break
                    train_loss = sum(train_losses)/counter
                    if not validate_each_epoch_cv:
                        print(f"Epoch {epoch} Mean Train Loss: {train_loss}")
                    if validate_each_epoch_cv:
                        with torch.no_grad():  # Disable gradient computation for inference
                            i = 0
                            counter = 0
                            val_losses = []
                            for X_batch, y_batch in val_loader:
                                counter += X_batch.shape[0]
                                outputs = model(X_batch)
                                if (i == 0) & (j % 5 == 0) & (j != 0) & (plot):
                                    plot_predictions(prediction_output=outputs, true=y_batch)
                                val_loss = criterion(outputs, y_batch)
                                val_losses.append(val_loss.detach().item())
                                i += 1
                            val_loss = sum(val_losses)/counter
                            print(f"Epoch {epoch}:  Mean Train Loss: {train_loss}, Mean Validation Loss: {val_loss}")
                    j += 1
                if start_over:
                    raise ValueError("Cross Validation Gradient Exploded")
                val_losses = []
                with torch.no_grad():  # Disable gradient computation for inference
                    i = 0
                    counter = 0
                    for X_batch, y_batch in val_loader:
                        counter += X_batch.shape[0]
                        outputs = model(X_batch)
                        if (i == 0) & (j != 0) & (plot):
                            plot_predictions(prediction_output=outputs, true=y_batch)
                        val_loss = criterion(outputs, y_batch)
                        val_losses.append(val_loss.detach().item())
                        i += 1
                val_loss = sum(val_losses)/counter
                print("----------------------------")
                print(f"Average Validation Loss For Fold {fold}: {val_loss}")
                print("----------------------------")
                overall_val_losses.append(val_loss)
            avg_validation_loss = sum(overall_val_losses) / num_splits
            nmix_scores.append(avg_validation_loss)
            print(f"Overall Average Accuracy for nmix == {nmix}: {avg_validation_loss}")
            print("-------------------------")
        min_index = nmix_scores.index(min(nmix_scores))
        nmix = nmixes[min_index]
        print(f"We select nmix = {nmix}")
        print("------------------------")

    # Train bagged models
    bag = 0
    while bag < 10:
        with open(f'data/tackle_image_bag_{bag}.pkl', 'rb') as f:
            tackle_dataset = pickle.load(f)
        
        print("**************---------------------*****************")
        print(f"Training Bag {bag} with N = {len(tackle_dataset)} observations.")
        print("**************---------------------*****************")

        # Create Data Loader
        train_dataloader = DataLoader(tackle_dataset, batch_size=512, shuffle=True)

        # Create Model
        model = BivariateGaussianMixture(nmix=nmix)

        # Define the optimizer (e.g., Stochastic Gradient Descent)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # Training loop
        num_epochs = 20
        
        losses = []
        start_over = False
        i = 0
        for epoch in range(num_epochs):
            j = 0
            train_losses = []
            counter = 0
            for X_batch, y_batch in train_dataloader:
                counter += X_batch.shape[0]
                optimizer.zero_grad()
                try:
                    output = model(X_batch)
                except:
                    print("Gradient Exploded, trying to train this model again...")
                    start_over = True
                    break
                if (j == 0) & (i % 5 == 0) & (plot):
                    plot_predictions(prediction_output=output, true=y_batch.detach().numpy())
                j+=1
                loss = criterion(output, y_batch)
                train_losses.append(loss.detach().item())
                # print(loss)
                loss.backward()
                optimizer.step()
            train_loss = sum(train_losses)/counter
            print(f"Epoch {epoch} Mean Train Loss : {train_loss}")
            i += 1
            if start_over:
                break
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
                i = 0
                with torch.no_grad():  # Disable gradient computation for inference
                    for X_batch, y_batch in test_dataloader:
                        counter += X_batch.shape[0]
                        outputs = model(X_batch)
                        if (i == 0) & (plot):
                            plot_predictions(prediction_output=outputs, true=y_batch)
                        test_loss = criterion(outputs, y_batch)
                        test_losses.append(test_loss.detach().item())
                test_loss = sum(test_losses)/counter
                records.append(pd.DataFrame({"bag" : [bag], "frames_from_eop" : [from_end_frame], "test_loss" : [test_loss]}))
                print("---------------------------------")
                print(f"Number of test samples for {from_end_frame} frames from EOP: {counter}")
                print(f"Test Loss for {from_end_frame} frames from EOP: {test_loss}")
        bag += 1
        torch.save(model.state_dict(), f'model_{bag}_weights.pth')

### Final ensemble on test
ensemble_model = TackleNetEnsemble(num_models=1, N=1, nmix = nmix)
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
        counter = 0
        for X_batch, y_batch in test_dataloader:
            counter += X_batch.shape[0]
            pred = ensemble_model.predict_pdf(X_batch)
            output_model = pred['mixture_return']
            outputs = pred['overall_pred']
            test_loss = criterion(output_model, y_batch)
            test_losses.append(test_loss.detach().item())
            if (i == 0) & (plot):
                plot_field(outputs, y_batch)
            i += 1
    test_loss = sum(test_losses)/counter
    records.append(pd.DataFrame({"bag" : ["ensemble"], "frames_from_eop" : [from_end_frame], "test_loss" : [test_loss]}))
    print("---------------------------------")
    print(f"Number of test samples for {from_end_frame} frames from EOP: {counter}")
    print(f"Test Loss for {from_end_frame} frames from EOP: {test_loss}")

records = pd.concat(records, axis = 0)
records.to_csv("test_loss_track.csv")
