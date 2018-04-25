import pandas as pd

train = pd.read_csv('../data/trainV_rfImp.csv')
val = pd.read_csv('../data/valV_rfImp.csv')
test = pd.read_csv('../data/testV_rfImp.csv')

print('R^2 (MonitorData ~ MAIACUS_Optical_Depth_047_Terra_Nearest4, train): ' + str((train.loc[:, ('MAIACUS_Optical_Depth_047_Terra_Nearest4', 'MonitorData')].corr()**2).iloc[1,0]))
print('R^2 (MonitorData ~ MAIACUS_Optical_Depth_047_Terra_Nearest4, val): ' + str((val.loc[:, ('MAIACUS_Optical_Depth_047_Terra_Nearest4', 'MonitorData')].corr()**2).iloc[1,0]))
print('R^2 (MonitorData ~ MAIACUS_Optical_Depth_047_Terra_Nearest4, test): ' + str((test.loc[:, ('MAIACUS_Optical_Depth_047_Terra_Nearest4', 'MonitorData')].corr()**2).iloc[1,0]))