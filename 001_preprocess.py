import pandas as pd
import numpy as np

players = pd.read_csv("data/nfl-big-data-bowl-2024/players.csv")
players = players[['nflId', 'weight', 'position']]
plays = pd.read_csv("data/nfl-big-data-bowl-2024/plays.csv")
plays = plays[['gameId', 'playId', 'ballCarrierId', 'passLength']]


for week in range(1, 10):
    print(f"Augmenting tracking for week {week}...")
    tracking = pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_week_{week}.csv")
    tracking.loc[tracking['playDirection'] == 'left', 'x'] = 120 - tracking['x']
    tracking.loc[tracking['playDirection'] == 'left', 'dir'] = (360 - tracking['dir']) % 360 # x y switched so use 360
    tracking.loc[tracking['x'] > 111, 'x'] = 111
    tracking.loc[tracking['x'] < 9, 'x'] = 9
    tracking.loc[tracking['x'] < 0, 'y'] = 0
    tracking.loc[tracking['y'] >= 53.3, 'y'] = 53.2
    tracking.loc[tracking['y'] < 0, 'y'] = 0
    tracking['week'] = week

    # remove unneccecary frames
    tracking_ball_remove_no_event = tracking.query("(displayName == 'football')").groupby(['playId', 'gameId']).filter(
        lambda x: (x['event'].isin(["pass_forward", "run", "handoff"])).sum() > 0)
    tracking_ball_remove_no_end = tracking_ball_remove_no_event.query("(displayName == 'football')").groupby(['playId', 'gameId']).filter(
        lambda x: (x['event'].isin(['pass_outcome_touchdown', 'tackle', 'touchdown', 'fumble', 'out_of_bounds', 'qb_slide'])).sum() > 0)
    tracking_ball_remove_no_end['is_end_event'] = tracking_ball_remove_no_end['event'].isin(
        ['pass_outcome_touchdown', 'tackle', 'touchdown', 'fumble', 'out_of_bounds', 'qb_slide']).astype(int)
    final_tracking = tracking_ball_remove_no_end[['gameId', 'playId', 'frameId']].drop_duplicates().merge(tracking, how = 'left', on = ['gameId', 'playId', 'frameId'])

    # add needed features

    current_positions = final_tracking.merge(players, on = "nflId", how = "left")
    current_positions = current_positions.merge(plays, on = ['gameId', 'playId'], how = "left")

    current_positions['type'] = current_positions['position'].apply(
        lambda x: "Offense" if x in ["QB", "TE", "WR", "G", "RB", "C", "FB", "T"] else "Defense")
    current_positions['type'] = current_positions.apply(lambda row: 'Ball' if pd.isna(row['nflId']) else row['type'], axis=1)
    current_positions.loc[current_positions.nflId == current_positions.ballCarrierId, 'type'] = "Carrier"
    current_positions['dir_rad'] = np.radians(current_positions['dir']) # fix degrees
    current_positions['Sy'] = current_positions['s'] * np.cos(current_positions['dir_rad']) # x y axis switched
    current_positions['Sx'] = current_positions['s'] * np.sin(current_positions['dir_rad'])
    current_positions['Ay'] = current_positions['a'] * np.cos(current_positions['dir_rad'])
    current_positions['Ax'] = current_positions['a'] * np.sin(current_positions['dir_rad'])

    current_positions.to_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv")

