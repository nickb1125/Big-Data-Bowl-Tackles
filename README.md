# Spatial Density Estimation for Tackle Success Using Deep Convolutional Gauccian-Mixture Ensembling 
#### Expected Yards Saved (EYS) & Percent Field Influence (PFI): Metrics for Tackler Efficiencies & Role Classification

# Main Goals

* **A shortcoming of the current state of sports analytics is failing to recognize the limitations of our data and the degree of prediction confidence we may have.** Limited data means limited and **varying confidence depending on the in-play situation**, and we should account and report for these varying intervals. From play-call-decision analytics to  metrics like those in the BDB, we should start reporting metric confidence. **We estimate variance in prediction by using ensemble model methods for our spatial densities.**

* Tackling encompasses more than big hits: **valuable tacking skills include any type of coersion that reduces the ball carriers yardage** by the end of the play. This can be direct, like the highlight hits you'll see on replay, or indirect like having positioning that manipulates the ball carrier into a worse route or out of bounds. We should have ways of measuring how direct a defenders influence is spatially, and how much he reduces yardage.

* **Spatial density estimation and the use of the subtraction method allows us to (1) where the tackler is coercing the ball carrier to go in the context of the rest of his team, and (2) how direct, or broad, his influence on the ball carrier is.**


# Code Map

### 0. objects.py

##### Play Class
The `play` class is designed to represent and analyze a football play. It includes methods for extracting information about the play, such as player movements, end-of-play location, and calculating features related to player density and movement.

##### Data Processing and Analysis Functions
The code contains functions for processing tracking data, calculating player movement features, and generating field density matrices based on player positions. These functions contribute to the analysis of player interactions during a play.

##### TackleNetEnsemble Class
This class represents an ensemble of deep learning models for predicting tackle outcomes. It loads pre-trained models and provides a method for predicting the probability distribution of tackle locations.

##### GaussianMixtureLoss and BivariateGaussianMixture Classes
These classes define a custom loss function (`GaussianMixtureLoss`) and a neural network model (`BivariateGaussianMixture`) for handling Gaussian mixture distributions. These components are used in the TackleNetEnsemble class.

##### TackleAttemptDataset Class
This class represents a dataset of football plays, including images, labels, and information about yards gained. It is likely used for training and evaluating the TackleNetEnsemble models.

##### Utility Functions
The code includes utility functions for converting arrays to dataframes, calculating expected values from multivariate distributions, and generating visualizations of football play outcomes.

##### Training and Loading Models
The code includes loading pre-trained models for the TackleNetEnsemble and initializes instances of the custom neural network model (`BivariateGaussianMixture`).


### 1. preprocess.py
```
python 001_preprocess.py
```

##### Read Data Files

- Utilizes pandas to read two CSV files:
  - "players.csv" and "plays.csv" from the "nfl-big-data-bowl-2024" dataset.
  - Selects specific columns ('nflId', 'weight', 'position') from the "players" dataset.
  - Selects columns ('gameId', 'playId', 'ballCarrierId', 'passLength') from the "plays" dataset.

##### Iterate Over Weekly Tracking Data

- Loops through tracking data for weeks 1 to 9.
- Adjusts coordinates and directions for plays with the direction 'left'.
- Cleans and filters tracking data based on x and y coordinates and specific events.

##### Feature Engineering

- Determines start and end events related to the football.
- Merges relevant data, including game and play information, creating a final tracking dataset.
- Adds additional features like player positions, play type, and directional components (Sx, Sy, Ax, Ay).

##### Data Export

- Exports augmented tracking data for each week to CSV files.
- Filename format: "tracking_a_week_{week}.csv".

### 2. imageset.py

```
python 002_imageset.py
```

##### Loading Base Data
   - Reads play and tracking data, combining relevant information.

##### Test Data Preparation
   - Creates a dataset for testing, considering specific criteria and saving it for future use.

##### Training and Validation Data Preparation
   - Randomly selects frames for each play, creating datasets for training and validation.

##### Weighted Occurrence Analysis
   - Analyzes the occurrence percentages of yards gained for potential weight adjustments in the model.

### 3. train.py

```
python 003_train.py
```

##### Data Loading and Preprocessing
   - Reads occurrence weights and test set information.
   - Defines the loss function, device, and other configurations.

##### Cross-validation for GMM Hyperparameter Tuning
   - Optionally performs cross-validation to determine the optimal number of mixtures (nmix) for the GMM.

##### Training GMM Models
   - Trains individual GMM models on different datasets (bags) using tackle attempt images.
   - Saves the weights of each model.

##### Ensemble Model Training
   - Creates an ensemble model for final predictions on the test set.
   - Evaluates and records the test loss for each frame from the end of play (EOP).

##### Performance Evaluation
   - Records and saves the test loss for each model and the ensemble.

### 4. metric_track.py

```
python 004_metric_track.py
```

##### Metrics Extraction
   - Gets play metrics for defensive players.
   - Iterates through plays, identifying defensive players and calculating their expected contribution, SOI, and DOI.
   - Records and prints metrics for each defensive player and frame.
   - Saves a dictionary containing contribution metrics.


### 4. predictor.py

```
python 005_predictor.py
```

**Note: Change playId and gameId at top of script to complete for different plays**

###### 3D Animation for Individual Defensive Player Contributions
   - Defines functions for updating frames and generating animation.
   - Creates a 3D animation for each defensive player's contributions.
   - Saves individual animations as GIFs.

##### 3D Animation for Overall Spatial Tackle Density
   - Defines functions for updating frames and generating animation.
   - Creates a 3D animation for overall spatial tackle density.
   - Saves the overall animation as a GIF.

### 4. analysis.Rmd

Basic analysis for submission results.