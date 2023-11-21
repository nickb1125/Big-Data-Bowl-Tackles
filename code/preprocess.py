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
    tracking.to_csv(f"data/nfl-big-data-bowl-2024/tracking_a_week_{week}.csv")