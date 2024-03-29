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
from objects import play, TackleNetEnsemble
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

print("Loading base data")
print("-----------------")

# Read and combine tracking data for all weeks
tracking = pd.concat([pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv") for week in range(1, 10)])
# Filter to only tracking plays meeting criteria
plays = pd.read_csv("data/nfl-big-data-bowl-2024/plays.csv")
plays = tracking[['gameId', 'playId']].drop_duplicates().merge(plays, how = 'left', on = ['gameId', 'playId'])
# Define pretrained model
model = TackleNetEnsemble(num_models = 5, N = 3, nmix=5)

# get play metrics
all_games = tracking.query("week in [9]").gameId.unique() 
all_plays = plays.query("(gameId in @all_games)")

print("Getting metrics")
print("-----------------")

all_contributions = []
all_player_dict = {id : {'game_id' : [], 'play_id' : [], 'estimated_contribution' : [], 'lower_contribution' : [], 'upper_contribution' : [],
                         'estimated_soi' : [], 'lower_soi' : [], 'upper_soi' : [],
                         'estimated_doi' : [], 'lower_doi' : [], 'upper_doi' : []} for id in tracking.nflId.unique()}
for index, play_row in tqdm(all_plays.iterrows(), total=len(all_plays), desc="Processing Plays"):
    play_object = play(play_row.gameId, play_row.playId, tracking)
    try:
        def_df = play_object.refine_tracking(frame_id = play_object.min_frame)["Defense"]
    except:
        continue
    def_df=def_df.query("position in ['DE', 'DT', 'NT', 'ILB', 'MLB', 'OLB']")
    def_ids = def_df.nflId.unique()
    original_all_pred = play_object.predict_tackle_distribution(model=model, without_player_id = 0)
    for id in def_ids:
        omit_predict_object = play_object.predict_tackle_distribution(model=model, without_player_id = id)
        metric_object = play_object.get_expected_contribution(model, original_predict_object=original_all_pred, w_omission_all_pred=omit_predict_object)
        lower_contriution, exp_contriution, upper_contriution = metric_object['contribution']  # for all frames
        print(f"{id} Expected contribution frame 1: {round(exp_contriution[0], 2)} 95% CI: [{round(lower_contriution[0], 2)}, {round(upper_contriution[0], 2)}]")
        lower_soi, exp_soi, upper_soi = metric_object['soi'] # for all frames
        print(f"{id} Expected SOI frame 1: {round(exp_soi[0], 2)} 95% CI: [{round(lower_soi[0], 2)}, {round(upper_soi[0], 2)}]")
        lower_doi, exp_doi, upper_doi = metric_object['doi'] # for all frames
        print(f"{id} Expected DOI frame 1: {round(exp_doi[0], 2)} 95% CI: [{round(lower_doi[0], 2)}, {round(upper_doi[0], 2)}]")
        all_contributions.extend(exp_contriution)



        all_player_dict[id]['estimated_contribution'].extend(exp_contriution)
        all_player_dict[id]['lower_contribution'].extend(lower_contriution)
        all_player_dict[id]['upper_contribution'].extend(upper_contriution)

        all_player_dict[id]['estimated_soi'].extend(exp_soi)
        all_player_dict[id]['lower_soi'].extend(lower_soi)
        all_player_dict[id]['upper_soi'].extend(upper_soi)

        all_player_dict[id]['estimated_doi'].extend(exp_doi)
        all_player_dict[id]['lower_doi'].extend(lower_doi)
        all_player_dict[id]['upper_doi'].extend(upper_doi)
        
        all_player_dict[id]['play_id'].extend(np.repeat(play_row.playId, len(exp_contriution)))
        all_player_dict[id]['game_id'].extend(np.repeat(play_row.gameId, len(exp_contriution)))
    print("-----------------------")
    print(f"Current mean contribution: {sum(all_contributions)/len(all_contributions)}")
    print(f"Current min contribution: {min(all_contributions)}")
    print(f"Current max contribution: {max(all_contributions)}")
    print("-----------------------")

with open(f"data/contribution_dict.pkl", f'wb') as outp:  # Overwrites any existing file.
    pickle.dump(all_player_dict, outp, pickle.HIGHEST_PROTOCOL)

contribution_df_list = []
for id in tqdm(all_player_dict.keys()):
    if len(all_player_dict[id]["game_id"]) == 0:
        continue
    player_df = pd.DataFrame(all_player_dict[id])
    player_df['nflId'] = id
    contribution_df_list.append(player_df)
contribution_df=pd.concat(contribution_df_list, axis=0).reset_index(drop=1)
contribution_df = contribution_df.merge(tracking[['nflId', 'displayName', 'position']].drop_duplicates(), on = ['nflId'], how='left')

contribution_df.to_csv("data/contribution_df_final.csv", index = False)


    