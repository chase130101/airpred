import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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
        self.least_by_nan = least_by_nan

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
        
        for iter in range(self.max_iter):
            for i in least_by_nan:

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
            imputed = new_imputed.copy()
            if iter >= 1:
                if self.gamma_[iter] >= self.gamma_[iter-1]:
                    self.num_iter = iter
                    self.estimators_ = self.estimators_[:self.num_iter]
                    break
                elif iter == self.max_iter-1:
                    self.num_iter = iter+1
        
        return self

    def transform(self, X):
        check_is_fitted(self, ['statistics_', 'estimators_', 'gamma_'])
        X = check_array(X, copy=True, dtype=np.float64, force_all_finite=False)
        if X.shape[1] != self.statistics_.shape[1]:
            raise ValueError('X has %d features per sample, expected %d'
                             % (X.shape[1], self.statistics_.shape[1]))

        X_nan = np.isnan(X)
        imputed = self.initial_imputer.fit_transform(X)
        
        for iter in range(self.num_iter):
            for i in self.least_by_nan:
                    
                X_s = np.delete(imputed, i, 1)
                y_nan = X_nan[:, i]

                X_unk = X_s[y_nan]
                    
                estimator_ = self.estimators_[iter][i]
                if len(X_unk) > 0:
                    imputed[y_nan, i] = estimator_.predict(X_unk)

        return imputed