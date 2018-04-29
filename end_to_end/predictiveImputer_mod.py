import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.metrics import r2_score

class PredictiveImputer(BaseEstimator, TransformerMixin):
    def __init__(self, max_iter=10, initial_strategy='mean', f_model='RandomForest'):
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.initial_imputer = Imputer(strategy=initial_strategy)
        self.f_model = f_model

    def fit(self, X, y=None, **kwargs):
        X = check_array(X, dtype=np.float64, force_all_finite=False)

        X_nan = np.isnan(X)
        least_by_nan = X_nan.sum(axis=0).argsort()
        num_nan_least_to_greatest = np.sort(X_nan.sum(axis=0))
        self.least_by_nan = least_by_nan
        self.num_nan_least_to_greatest = num_nan_least_to_greatest
        
        imputed = self.initial_imputer.fit_transform(X)
        new_imputed = imputed.copy()

        self.statistics_ = np.ma.getdata(X)
        self.gamma_ = []

        if self.f_model == 'RandomForest':                                             
            self.estimators_ = [[RandomForestRegressor(**kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
        if self.f_model == 'Ridge':                                             
            self.estimators_ = [[Ridge(**kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
        elif self.f_model == 'KNN':
            self.estimators_ = [[KNeighborsRegressor(n_neighbors=min(5, sum(~X_nan[:, i])), **kwargs) for i in range(X.shape[1])] for j in range(self.max_iter)]
        
        print('Number of variables: ' + str(len(least_by_nan)))
        for iter in range(self.max_iter):
            print('Iteration ' + str(iter+1))
            var_counter = 0
            for i in least_by_nan:
                print('Variable # ' + str(var_counter))
                var_counter += 1

                X_s = np.delete(new_imputed, i, 1)
                y_nan = X_nan[:, i]

                X_train = X_s[~y_nan]
                y_train = new_imputed[~y_nan, i]
                X_unk = X_s[y_nan]

                estimator_ = self.estimators_[iter][i]
                estimator_.fit(X_train, y_train)
                if len(X_unk) > 0:
                    new_imputed[y_nan, i] = estimator_.predict(X_unk)
                    
            gamma = np.sum((new_imputed-imputed)**2)/np.sum(new_imputed**2)
            self.gamma_.append(gamma)
            print('Difference: ' + str(gamma))
            print()
            imputed = new_imputed.copy()
            if iter >= 1:
                if self.gamma_[iter] >= self.gamma_[iter-1]:
                    self.num_iter = iter
                    self.estimators_ = self.estimators_[:self.num_iter]
                    break
                elif iter == self.max_iter-1:
                    self.num_iter = iter+1
        
        return self

    def transform(self, X, evaluate = True, backup_impute_strategy = 'mean'):
        check_is_fitted(self, ['statistics_', 'estimators_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError('X has %d features per sample, expected %d'
                             % (X.shape[1], self.statistics_.shape[1]))

        X_nan = np.isnan(X)
        imputed = self.initial_imputer.fit_transform(X)
        
        if evaluate == True:
            r2_numNaN_matrix = np.zeros((imputed.shape[1], 2))
        for iter in range(self.num_iter):
            for i, num_nan in zip(self.least_by_nan, self.num_nan_least_to_greatest):
                    
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]

                X_unk = X_s[y_nan]
                    
                estimator_ = self.estimators_[iter][i]
                if len(X_unk) > 0:
                    imputed[y_nan, i] = estimator_.predict(X_unk)
            
                X_known = X_s[~y_nan]
                y_known = imputed[~y_nan, i]
                pred = estimator_.predict(X_known)
                r2 = r2_score(y_known, pred)

                if r2 < 0:
                    backup_imputer = Imputer(strategy = backup_impute_strategy)
                    backup_imputer.fit(y_known.reshape(y_known.shape[0], -1))
                    if len(X_unk) > 0:
                        imputed[y_nan, i] = np.repeat(backup_imputer.statistics_, imputed[y_nan, i].shape[0])
                    pred = np.repeat(backup_imputer.statistics_, y_known.shape[0])
                    r2 = r2_score(y_known, pred)

                if iter == self.num_iter-1 and evaluate == True:
                    r2_numNaN_matrix[i, 0] = r2
                    r2_numNaN_matrix[i, 1] = num_nan

        if evaluate == True:
            r2_numNaN_df = pd.DataFrame(r2_numNaN_matrix, columns = ['R2', 'num_missing'])
            self.r2_scores_df = r2_numNaN_df
            return imputed, r2_numNaN_df
        else:
            return imputed