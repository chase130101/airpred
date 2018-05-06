"""Description: This is a class for making model-based data imputations that is inspired by
the MissForest algorithm (see: https://academic.oup.com/bioinformatics/article/28/1/112/219101). The code
has been adapted from the following GitHub repo: https://github.com/log0ymxm/predictive_imputer. Significant
changes were made to the transform method and the stopping criterion used in the fit method in order to better
match the MissForest algorithm from the original paper.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import r2_score

class PredictiveImputer(BaseEstimator, TransformerMixin):
    """Allows the user to fit data imputation models and return imputed data matrices
    - Inspired by MissForest data imputation algorithm
    -----------
    Methods:
        - fit: Fits data imputation models
        - transform: Imputes data matrices using fitted data imputation models
        -- transform can only be called following a call to fit
        -- Data matrix to impute with transform must have the same number of columns as the
        data matrix used when fitting the data imputation models
    -----------
    Attributes user can specify when instantiating class:
        - max_iter (int): Maximium number of iterations to run MissForest algorithm for; default is 10
        - initial_strategy (str): Must be one of 'mean' or 'median'; how to impute data matrices prior
        to the first iteration of the MissForest algorithm; default is 'mean'
        - f_model (str): Must be one of 'Ridge' or 'RandomForest'; modeling method to use for data imputation;
        default is 'RandomForest'
    -----------
    Other attributes upon class instantiation:
        - initial_imputer (sklearn.preprocessing.Imputer): For imputing data matrices prior to first iteration
        of MissForest algorithm
    -----------
    New attributes following a call to fit:
        - least_by_nan (list of int): Column indices sorted from least missing to most missing in data matrix
        passed to fit
        - gamma_ (list of float): Values used for computing stopping criterion values
        - estimators_ (list of sklearn models): Fitted models that will be used to impute data matrices upon
        calls to transform
        - statistics_ (numpy.array): Used for determining if data matrix passed to transform has
        same number of columns as data matrix previously passed to fit
    -----------
    New attributes following a call to transform:
        - r2_scores_df (pandas.DataFrame): First column is R^2 for imputations of each variable and 
        second column is number of missing values for each variable in data matrix passed to transform
    """
    def __init__(self, max_iter=10, initial_strategy='mean', f_model='RandomForest'):
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.initial_imputer = Imputer(strategy=initial_strategy)
        self.f_model = f_model

    def fit(self, X, **kwargs):
        """Fits data imputation models that will be used when transform is called to impute data matrices
        -----------
        Inputs:
            - X (numpy.array or pandas.DataFrame): Data matrix to fit imputation models on
            -- All columns must have at least one non-missing value
            -- X.shape[1] must be > 1
            - **kwargs: Hyper-parameters for imputation models
            -- Must be valid argument names/values for the modeling method of choice ('Ridge' or 'RandomForest')
            -- Default arguments for modeling method will be used if no arguments are specified
        -----------
        Outputs:
            self: Includes everything necessary for calling transform method
        """
        X = check_array(X, dtype=np.float64, force_all_finite=False)

        # determine where nans are in data and find number of nans in each column
        # impute columns with fewest nans first in each iteration
        X_nan = np.isnan(X)
        least_by_nan = X_nan.sum(axis=0).argsort()
        self.least_by_nan = least_by_nan
                
        # impute mean of each column for first iteration
        imputed = self.initial_imputer.fit_transform(X)
        new_imputed = imputed.copy()

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []
        
        # different modeling methods to use; ultimately, one model will be fitted per column
        if self.f_model == 'RandomForest':                                             
            self.estimators_ = [RandomForestRegressor(**kwargs) for i in range(X.shape[1])]
        if self.f_model == 'Ridge':                                             
            #self.estimators_ = [[Ridge(**kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
            self.estimators_ = [Ridge(**kwargs) for i in range(X.shape[1])]
        
        print('Number of variables: ' + str(len(least_by_nan)))
        for iter in range(self.max_iter):
            print('Iteration ' + str(iter+1))
            var_counter = 0
            for i in least_by_nan:
                print('Variable # ' + str(var_counter))
                var_counter += 1
                
                # delete column to impute from X; get nan indicators for column to impute
                X_s = np.delete(new_imputed, i, 1)
                y_nan = X_nan[:, i]
                
                # train using rows for y is not nan; get X rows where y is nan
                X_train = X_s[~y_nan]
                y_train = new_imputed[~y_nan, i]
                X_unk = X_s[y_nan]
                
                estimator_ = self.estimators_[i]
                estimator_.fit(X_train, y_train)
                if len(X_unk) > 0:
                    # fill in values for which y is nan with model imputations
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
        - Also returns a dataframe with an R^2 evaluation for each imputation model and the 
        number of missing values for each variable if evaluate set to True
        - Can only be called following a call to fit
        -----------
        Inputs:
            - X (np.array or pandas.DataFrame): Data matrix to be imputed
            -- All columns must have at least one non-missing value
            -- X.shape[1] must be the same shape 
            - evaluate (boolean): Whether or not to evaluate the imputations
            - backup_impute_strategy (str): Must be one of 'mean' or 'median'; method to make imputations when 
            an imputation model performs poorly (R^2 < 0)                             
        -----------
        Outputs:
             - imputed (numpy.array): Imputed data matrix
             - r2_numNaN_df (pandas.DataFrame): Only if evaluate set to True; first column is R^2 for imputations 
             of each variable and second column is number of missing values for each variable in data matrix 
             passed to transform
        """
        check_is_fitted(self, ['statistics_', 'estimators_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError('X has %d features per sample, expected %d' % (X.shape[1], self.statistics_.shape[1]))

        # determine where nans are in data
        X_nan = np.isnan(X)
        
        # impute mean of each column for first iteration
        imputed = self.initial_imputer.fit_transform(X)
        
        if evaluate == True:
            r2_numNaN_matrix = np.zeros((imputed.shape[1], 2))
        for iter in range(self.num_iter):
            for i in self.least_by_nan:
                    
                # delete column to impute from X; get nan indicators for column to impute
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]
                
                # get X rows where y is nan
                X_unk = X_s[y_nan]
                    
                estimator_ = self.estimators_[i]
                if len(X_unk) > 0:
                    # fill in values for which y is nan with model imputations
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
                
                # save R^2 for each column in final iteration and number of nan for each column
                if iter == self.num_iter-1 and evaluate == True:
                    r2_numNaN_matrix[i, 0] = r2
                    r2_numNaN_matrix[i, 1] = np.sum(y_nan)
        
        # return imputed matrix and 
        # (if evaluate set to True) dataframe where first column is R^2 for imputations of each variable and 
        # second column is number of missing values for each variable
        if evaluate == True:
            r2_numNaN_df = pd.DataFrame(r2_numNaN_matrix, columns = ['R2', 'num_missing'])
            self.r2_scores_df = r2_numNaN_df
            return imputed, r2_numNaN_df
        else:
            return imputed