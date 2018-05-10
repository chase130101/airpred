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
             'test'                  : '../data/test_rfImp.csv',
             'trainV'                : '../data/trainV_rfImp.csv',
             'valV'                  : '../data/valV_rfImp.csv',
             'testV'                 : '../data/testV_rfImp.csv',
             'r2_scores'             : '../data/r2_scores_rfImp.csv',
}


config["Ridge_Imputation"] = {
             'train'                 : '../data/train_ridgeImp.csv',
             'test'                  : '../data/test_ridgeImp.csv',
             'trainV'                : '../data/trainV_ridgeImp.csv',
             'valV'                  : '../data/valV_ridgeImp.csv',
             'testV'                 : '../data/testV_ridgeImp.csv',
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
             'rf_fti'                : '../data/rf_feature_importances.csv',
             'xgb_pred'              : '../data/test_xgboostPred.csv',
             'xgb_fti'               : '../data/xgboost_feature_importances.csv',      
}

config["CNN_hyperparam_1"] = { 
             'hidden_size_conv'      : 50,
             'kernel_size'           : 5,
             'padding'               : 1,
             'hidden_size_full'      : 100,
             'dropout_full'          : 0.1,
             'hidden_size_combo'     : 100,
             'dropout_combo'         : 0.1,
             'lr'                    : 0.1,
             'weight_decay'          : 0.00001
}

config["CNN_hyperparam_2"] = { 
             'hidden_size_conv'      : 25,
             'kernel_size'           : 3,
             'padding'               : 1,
             'hidden_size_full'      : 100,
             'dropout_full'          : 0.1,
             'hidden_size2_full'     : 100,
             'dropout2_full'         : 0.1,
             'lr'                    : 0.1,
             'weight_decay'          : 0.00001
}


with open('py_config.ini', 'w') as configfile:
    config.write(configfile)