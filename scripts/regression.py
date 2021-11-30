

from typing import Tuple
import json


import matplotlib.pyplot as plt
import os.path
import pandas as pd

from sklearn.utils._testing import ignore_warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score

def do_regression(X_train, y_train, X_test, y_test):

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


    best_training_score = 0
    best_index = None

    results = []

    for name, m in models.items():
        print('Training with model: {}'.format(name))
        grid = GridSearchCV(estimator=m, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)


        if best_training_score < grid_result.best_score_:
            best_index = len(results)

        y_pred = grid_result.predict(X_test)

        results.append({
            'Classifier': name,
            'Training Score': grid_result.best_score_,
            'Parameters': grid_result.best_params_,
            'Test Score': r2_score(y_test, y_pred)
        })

    return results, best_index

if __name__ == '__main__':

    datafile = os.path.abspath(os.path.dirname(__file__)) + "/../data/clean_data.csv"
    df = pd.read_csv(datafile, skipinitialspace=True)
    df = df.select_dtypes(include='number')
    df = df.fillna(df.mean())

    X = df.loc[:, df.columns != 'Salary'].to_numpy()
    y = df['Salary'].to_numpy()

    Xscaler = StandardScaler().fit(X)
    X = Xscaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20)

    print("Running the regression...\n")
    results, best_index = do_regression(X_train, y_train, X_test, y_test)

    print("...done\n")

    print("Results:\n")
    for r in results:
        print(json.dumps(r, indent=4, sort_keys=True))
        print()

    print("Best performer: \n")
    print(json.dumps(results[best_index], indent=4, sort_keys=True))
    print()
    

