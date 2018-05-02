import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn.metrics
import pickle
from data_split_tune_utils import X_y_site_split

#train = pd.read_csv('../data/train_ridgeImp.csv')
#test = pd.read_csv('../data/test_ridgeImp.csv')
train = pd.read_csv('../data/train_rfImp.csv')
test = pd.read_csv('../data/test_rfImp.csv')

# drop rows with no monitor data response value
train = train.dropna(axis=0)
test = test.dropna(axis=0)

# split datasets into x, y, and site
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

# load dictionary with best hyper-parameters from cross-validation
best_hyperparams = pickle.load(open('best_xgboost_hyperparams.pkl', 'rb'))

# model to fit
xgboost = xgb.XGBRegressor(random_state=1, n_jobs=-1)

# set model attributes to best combination of hyper-parameters from cross-validation
for key in list(best_hyperparams.keys()):
    setattr(xgboost, key, best_hyperparams[key])

# fit model on train data and make predictions on test data; compute test R^2
xgboost.fit(train_x, train_y)
test_pred_xgboost = xgboost.predict(test_x)
test_r2_xgboost = sklearn.metrics.r2_score(test_y, test_pred_xgboost)
print('Test R^2: ' + str(test_r2_xgboost))

# put model predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
test['MonitorData_pred'] = pd.Series(test_pred_xgboost, index=test.index)

# save test dataframe with predictions
test.to_csv('../data/test_xgboostPred.csv', index=False)
#pickle.dump(xgboost, open('xgboost_final.pkl', 'wb'))

# create dataframe of feature importances and save
feature_importance_df = pd.DataFrame(xgboost.feature_importances_.reshape(len(xgboost.feature_importances_), -1), columns=['XGBoost_Feature_Importance'])
feature_importance_df['Variable'] = pd.Series(train_x.columns, index=feature_importance_df.index)
feature_importance_df.to_csv('../data/xgboost_feature_importances.csv', index=False)
