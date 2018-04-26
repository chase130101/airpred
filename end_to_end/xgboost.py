import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn as sk
#from matplotlib import pyplot

data = pd.read_csv('../data/imputed_data.csv')

def train_test_split(data, train_prop, site_var_name='site'):
    """Splits data into train test sets by PM2.5 monitor site
    
    Arguments:
        data (pandas dataframe): Data to be split
        site_var_name (string): Site ID variable name
        train_prop (float): Proportion of sites to be put into train set
    """
    # get sites for train data
    train_sites = np.random.choice(np.unique(data[site_var_name].values), round(len(np.unique(data[site_var_name].values))*train_prop), replace = False)
        
    # get train and test
    train = data[data[site_var_name].isin(train_sites)]
    test = data[~data[site_var_name].isin(train_sites)]
    
    return train, test

def X_y_site_split(data, y_var_name='MonitorData', site_var_name='site'):
    """Splits a dataframe into X, y, site ID
    
    Arguments:
        data (pandas dataframe): Data to be split
        y_var_name (string): Response variable name
        site_var_name (string): Site ID variable name
    """
    data_y = data.loc[:, y_var_name]
    data_sites = data.loc[:, site_var_name]
    data_x = data.drop([y_var_name, site_var_name], axis=1)
    
    return data_x, data_y, data_sites

train, test = train_test_split(data, train_prop = .8)

x_train, y_train, sites_train = X_y_site_split(train)
x_test, y_test, sites_test = X_y_site_split(test)

param = {'learning_rate':.1, 'max_depth':6, 'n_estimators':100}

#Information on gradient boosting parameter tuning
#https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
#param = {'learning_rate':[.001, .01, .05, .1], 'max_depth':[4, 6, 8, 10], 'n_estimators':[100, 250, 500, 750, 1000]}

model = xgb.XGBRegressor(**param)
model.fit(x_train, y_train)
print(model)

y_pred = model.predict(x_test)

rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, y_pred))
print("RMSE: %.4f" % rmse)

r_sq = sk.metrics.r2_score(y_test, y_pred)
print("R-squared: %.4f" % r_sq)

#Variable importance plot
#xgb.plot_importance(model, max_num_features=20)
#pyplot.show()
