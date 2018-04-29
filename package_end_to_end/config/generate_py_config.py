# To run this, simple run "python3 generate_py_config.py" in a terminal

import configparser

config = configparser.RawConfigParser()

config["RF_Train_Test_Imputation"] = {
             'data_to_impute_path'   : '../data/data_to_impute.csv',
             'train_imputed_rf_path' : '../data/train_rfImp.csv',
             'test_imputed_rf_path'  : '../data/test_rfImp.csv',
             'rf_impute_model'       : 'rf_imputer.pkl'
}


config["RF_Train_Test_Val_Imputation"] = {
             'data_to_impute_path'   : '../data/data_to_impute.csv',
             'train_imputed_rf_path' : '../data/trainV_rfImp.csv',
             'val_imputed_rf_path'   : '../data/valV_rfImp.csv',
             'test_imputed_rf_path'  : '../data/testV_rfImp.csv',
             'rf_impute_model'       : 'rfV_imputer.pkl'
}


config["Ridge_Train_Test_Imputation"] = {
             'data_to_impute_path'      : '../data/data_to_impute.csv',
             'train_imputed_ridge_path' : '../data/train_ridgeImp.csv',
             'test_imputed_ridge_path'  : '../data/test_ridgeImp.csv',
             'ridge_impute_model'       : 'ridge_imputer.pkl'
}


config["Ridge_Train_Test_Val_Imputation"] = {
             'data_to_impute_path'      : '../data/data_to_impute.csv',
             'train_imputed_ridge_path' : '../data/trainV_ridgeImp.csv',
             'val_imputed_ridge_path'   : '../data/valV_ridgeImp.csv',
             'test_imputed_ridge_path'  : '../data/testV_ridgeImp.csv',
             'rf_impute_model'          : 'ridgeV_imputer.pkl'
}


with open('py_config.ini', 'w') as configfile:
    config.write(configfile)
