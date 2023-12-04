import pandas as pd

for week in range(1, 10):
    print(f"Augmenting tracking for week {week}...")
    tracking = pd.read_csv(f"data/nfl-big-data-bowl-2024/tracking_week_{week}.csv")
    tracking.loc[tracking['x'] > 120, 'x'] = 120
    tracking.loc[tracking['x'] < 0, 'y'] = 0
    tracking.loc[tracking['y'] > 53.3, 'y'] = 53.3
    tracking.loc[tracking['y'] < 0, 'y'] = 0
    tracking.loc[tracking['playDirection'] == 'left', 'x'] = 120 - tracking['x']
    tracking.loc[tracking['playDirection'] == 'left', 'dir'] = (180 - tracking['dir']) % 360
    tracking['week'] = week

    # remove unneccecary frames
    tracking_ball_remove_no_event = tracking.query("(displayName == 'football')").groupby(['playId', 'gameId']).filter(lambda x: (x['event'].isin(["pass_outcome_caught",
                                                                        "run", "handoff"])).sum() > 0)
    tracking_ball_remove_no_event['is_start_event'] = tracking_ball_remove_no_event['event'].isin(["pass_outcome_caught", "run", "handoff"]).astype(int)
    no_pre_event = tracking_ball_remove_no_event[tracking_ball_remove_no_event.groupby(['playId', 'gameId'])['is_start_event'].cumsum().eq(1)]
    tracking_ball_remove_no_end = no_pre_event.query("(displayName == 'football')").groupby(['playId', 'gameId']).filter(lambda x: (x['event'].isin(
        ['pass_outcome_touchdown', 'tackle', 'touchdown', 'fumble', 'out_of_bounds', 'qb_slide'])).sum() > 0)
    tracking_ball_remove_no_end['is_end_event'] = tracking_ball_remove_no_end['event'].isin(
        ['pass_outcome_touchdown', 'tackle', 'touchdown', 'fumble', 'out_of_bounds', 'qb_slide']).astype(int)
    no_post_end = tracking_ball_remove_no_end[tracking_ball_remove_no_end.groupby(['playId', 'gameId'])['is_end_event'].cumsum().eq(0)]

    no_post_end.to_csv(f"data/nfl-big-data-bowl-2024/tracking_no_aug_week_{week}.csv")