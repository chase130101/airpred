# To run this, simple run "python3 generate_py_config.py" in a terminal

import configparser

config = configparser.RawConfigParser()


config["data"] = {
             'train'                : '../data/train.csv',
             'test'                 : '../data/test.csv',
             'trainV'               : '../data/trainV.csv',
             'valV'                 : '../data/valV.csv',
             'testV'                : '../data/testV.csv',
             'data_to_impute'       : '../data/data_to_impute.csv',      
}


config["RF_Imputation"] = {
             'train'                 : '../data/train_rfImp.csv',
             'test'                  : '../data/val_rfImp.csv',
             'val'                   : '../data/test_rfImp.csv',
             'r2_scores'             : '../data/r2_scores_rfImp.csv',
}


config["Ridge_Imputation"] = {
             'train'                 : '../data/train_ridgeImp.csv',
             'test'                  : '../data/test_ridgeImp.csv',
             'val'                   : '../data/val_ridgeImp.csv',
             'r2_scores'             : '../data/r2_scores_ridgeImp.csv',
             'model'                 : 'ridge_imputer.pkl'
}

config["Reg_Best_Hyperparams"] = {
             'ridge'                 : 'best_ridge_hyperparams.pkl',
             'rf'                    : 'best_rf_hyperparams.pkl',
             'xgb'                   : 'best_xgboost_hyperparams.pkl',  
}

config["Regression"] = {
             'ridge_pred'            : '../data/test_ridgePred.csv',
             'ridge_final'           : 'ridge_final.pkl',
             'rf_pred'               : '../data/test_rfPred.csv',
             'rf_ft'                 : '../data/rf_feature_importances.csv',
             'xgb_pred'              : '../data/test_xgboostPred.csv',
             'xgb_ft'                : '../data/xgboost_feature_importances.csv',
      
}


with open('py_config.ini', 'w') as configfile:
    config.write(configfile)