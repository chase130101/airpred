import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import r2_score

### Inspired by MissForest imputation method
class PredictiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=10, initial_strategy='mean', f_model='RandomForest'):
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.initial_imputer = Imputer(strategy=initial_strategy)
        self.f_model = f_model

    def fit(self, X, **kwargs):
        """Returns is a list of fitted imputation models and other values that may be used when
        making imputations on a new datasets
        
        Arguments:
            X (matrix or pandas.DataFrame): For fitting imputation models on (all columns must have at least one non-missing value)
            **kwargs: Hyper-parameters for imputation models
        """
        X = check_array(X, dtype=np.float64, force_all_finite=False)

        # determine where NaNs are in data and find number of NaNs in each column
        # impute columns with fewest NaNs first in each iteration
        X_nan = np.isnan(X)
        least_by_nan = X_nan.sum(axis=0).argsort()
        self.least_by_nan = least_by_nan
                
        # impute mean of each column for first iteration
        imputed = self.initial_imputer.fit_transform(X)
        new_imputed = imputed.copy()

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []
        
        # different modeling methods to use; one model per column
        if self.f_model == 'RandomForest':                                             
            #self.estimators_ = [[RandomForestRegressor(**kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
            self.estimators_ = [RandomForestRegressor(**kwargs) for i in range(X.shape[1])]
        if self.f_model == 'Ridge':                                             
            #self.estimators_ = [[Ridge(**kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
            self.estimators_ = [Ridge(**kwargs) for i in range(X.shape[1])]
        elif self.f_model == 'KNN':
            #self.estimators_ = [[KNeighborsRegressor(n_neighbors=min(5, sum(~X_nan[:, i])), **kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
            self.estimators_ = [KNeighborsRegressor(n_neighbors=min(5, sum(~X_nan[:, i])), **kwargs) for i in range(X.shape[1])]
        
        print('Number of variables: ' + str(len(least_by_nan)))
        for iter in range(self.max_iter):
            print('Iteration ' + str(iter+1))
            var_counter = 0
            for i in least_by_nan:
                print('Variable # ' + str(var_counter))
                var_counter += 1
                
                # delete column to impute from X; get NaN indicators for column to impute
                X_s = np.delete(new_imputed, i, 1)
                y_nan = X_nan[:, i]
                
                # train using rows for y is not NaN; get X rows where y is NaN
                X_train = X_s[~y_nan]
                y_train = new_imputed[~y_nan, i]
                X_unk = X_s[y_nan]
                
                estimator_ = self.estimators_[i]
                estimator_.fit(X_train, y_train)
                if len(X_unk) > 0:
                    # fill in values for which y is NaN with model imputations
                    new_imputed[y_nan, i] = estimator_.predict(X_unk)
            
            # value used in stopping criterion
            gamma = np.sum((new_imputed-imputed)**2)/np.sum(new_imputed**2)
            self.gamma_.append(gamma)
            print('Difference: ' + str(gamma))
            print()
            
            if iter >= 1:
                if self.gamma_[iter] >= self.gamma_[iter-1]: # stopping criterion
                    self.num_iter = iter
                    
                    # use fitted models from previous iteration as final models
                    self.estimators_ = list(old_estimators) 
                    break
                elif iter == self.max_iter-1: # reached max iterations (alternative stopping criterion)
                    self.num_iter = iter+1
                    break
                    
            imputed = new_imputed.copy()
            old_estimators = tuple(self.estimators_)
        
        return self

    def transform(self, X, evaluate=True, backup_impute_strategy='mean'):
        """Returns an impuated data matrix 
        Also returns a dataframe with an R^2 evaluation for each imputation model and the 
        number of missing values for each variable if evaluate set to True
        
        Arguments:
            X (matrix or pandas.DataFrame): Data matrix to be imputed (all columns must have at least one non-missing value)
            evaluate (boolean): Whether or not to evaluate the imputations
            backup_impute_strategy (str): Method to make imputations when an imputation model performs poorly (R^2 < 0)
                                          (must be one of 'mean' or 'median')
        """
        check_is_fitted(self, ['statistics_', 'estimators_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError('X has %d features per sample, expected %d' % (X.shape[1], self.statistics_.shape[1]))

        # determine where NaNs are in data
        X_nan = np.isnan(X)
        
        # impute mean of each column for first iteration
        imputed = self.initial_imputer.fit_transform(X)
        
        if evaluate == True:
            r2_numNaN_matrix = np.zeros((imputed.shape[1], 2))
        for iter in range(self.num_iter):
            for i in self.least_by_nan:
                    
                # delete column to impute from X; get NaN indicators for column to impute
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]
                
                # get X rows where y is NaN
                X_unk = X_s[y_nan]
                    
                estimator_ = self.estimators_[i]
                if len(X_unk) > 0:
                    # fill in values for which y is NaN with model imputations
                    imputed[y_nan, i] = estimator_.predict(X_unk)
                
                # get rows where y is known; make predictions and compute R^2
                X_known = X_s[~y_nan]
                y_known = imputed[~y_nan, i]
                pred = estimator_.predict(X_known)
                r2 = r2_score(y_known, pred)
                
                # if model predictions are expected to be worse than mean imputation
                # use backup imputation strategy (mean or median)
                if r2 < 0: 
                    backup_imputer = Imputer(strategy=backup_impute_strategy)
                    backup_imputer.fit(y_known.reshape(y_known.shape[0], -1))
                    if len(X_unk) > 0:
                        imputed[y_nan, i] = np.repeat(backup_imputer.statistics_, imputed[y_nan, i].shape[0])
                        
                    # using backup imputation strategy compute new R^2 (should be zero if backup is mean)
                    pred = np.repeat(backup_imputer.statistics_, y_known.shape[0])
                    r2 = r2_score(y_known, pred)
                
                # save R^2 for each column in final iteration and number of NaNs for each column
                if iter == self.num_iter-1 and evaluate == True:
                    r2_numNaN_matrix[i, 0] = r2
                    r2_numNaN_matrix[i, 1] = np.sum(y_nan)
        
        # return imputed matrix and 
        # dataframe where first column is R^2 for imputations of each variable and 
        # second column is number of missing values for each variable (if evaluate==True)
        if evaluate == True:
            r2_numNaN_df = pd.DataFrame(r2_numNaN_matrix, columns = ['R2', 'num_missing'])
            self.r2_scores_df = r2_numNaN_df
            return imputed, r2_numNaN_df
        else:
            return imputed