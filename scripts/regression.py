from typing import Tuple
import math 

import matplotlib.pyplot as plt
import os.path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn import metrics


if __name__ == '__main__':
    datafile = os.path.abspath(os.path.dirname(__file__)) + "/../data/clean_data.csv"
    df = pd.read_csv(datafile, skipinitialspace=True)

    X = df.loc[:, df.columns != 'Salary'].to_numpy()
    y = df['Salary'].to_numpy()

    Xscaler = StandardScaler().fit(X)
    X = Xscaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20)


    alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    ls = Lasso()
    en = ElasticNet()
    r = Ridge()

    models = {
        "Lasso": ls,
        "ElasticNet": en,
        "Ridge": r
    }

    param_grid = dict(alpha=alpha)

    for name, m in models.items():
        print('Training with model: {}'.format(name))
        grid = GridSearchCV(estimator=m, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)

        print(grid_result.best_score_)


