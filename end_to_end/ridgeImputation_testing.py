import pandas as pd

train = pd.read_csv('../data/train_ridgeImp.csv')
test = pd.read_csv('../data/test_ridgeImp.csv')

print('R^2 (MonitorData ~ MAIACUS_Optical_Depth_047_Terra_Nearest4, train): ' + str((train.loc[:, ('MAIACUS_Optical_Depth_047_Terra_Nearest4', 'MonitorData')].corr()**2).iloc[1,0]))
print('R^2 (MonitorData ~ MAIACUS_Optical_Depth_047_Terra_Nearest4, test): ' + str((test.loc[:, ('MAIACUS_Optical_Depth_047_Terra_Nearest4', 'MonitorData')].corr()**2).iloc[1,0]))

print('R^2 (MonitorData ~ Nearby_Peak2_PM2.5, train): ' + str((train.loc[:, ('Nearby_Peak2_PM25', 'MonitorData')].corr()**2).iloc[1,0]))
print('R^2 (MonitorData ~ Nearby_Peak2_PM2.5, test): ' + str((test.loc[:, ('Nearby_Peak2_PM25', 'MonitorData')].corr()**2).iloc[1,0]))
