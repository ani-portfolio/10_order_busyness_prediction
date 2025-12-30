from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def split_data(df:pd.DataFrame, features: list, target: list, test_size: float, random_state: float):

    """
    
    """

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.DataFrame, param_grid: dict, cv_folds: int, cv_scoring: str, regressor_n_jobs: int, cv_n_jobs:int, random_state: int) -> BaseEstimator:
    """
    
    """

    regr = RandomForestRegressor(random_state=random_state, n_jobs=regressor_n_jobs)
    grid_search = GridSearchCV(estimator=regr,
                            param_grid=param_grid,
                            cv=cv_folds,
                            n_jobs=cv_n_jobs, verbose=1, scoring=cv_scoring)

    logger.info('Grid search best params')
    grid_search.fit(X_train, y_train.values.ravel()) #TODO Added ravel to fix warning (A column-vector y was passed when a 1d) 

    return grid_search

def train_refitted_model(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, best_params: dict) -> BaseEstimator:
    """
    
    """

    refitted_model = RandomForestRegressor(**best_params)

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    logger.info('Re-fit model on all data using best params')
    refitted_model.fit(X_all, y_all.values.ravel())

    return refitted_model

