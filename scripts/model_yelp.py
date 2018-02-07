import logging
import pprint

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

import fooodspace as fs


def tune_estimator(data, y, save_checkpoint=True):
    """Construct estimator, tune its hyperparameters, return gridsearch."""
    est = make_pipeline(
        fs.build_feature_union(),
        # LinearRegression(),
        fs.EnsembleEstimator(
            # baseest=LinearRegression(),
            baseest=Ridge(),
            # resiest=KNeighborsRegressor(),
            resiest=RandomForestRegressor(n_jobs=1),
        ),
    )

    g = GridSearchCV(
        est,
        {
            # Ridge
            'ensembleestimator__baseest__alpha': np.logspace(-4, 0.5, 8),
            # 'ensembleestimator__baseest__alpha': [.02, .025, .03, .035, .04],
            # KNeighborsRegressor:
            # 'ensembleestimator__resiest__n_neighbors': range(4, 7),
            # RandomForestRegressor:
            # 'ensembleestimator__resiest__min_samples_leaf': [1, 5, 10, 15, 20, 25],
            # 'ensembleestimator__resiest__min_samples_leaf': [4, 5, 6, 7, 8],
            'ensembleestimator__resiest__n_estimators': [100, 150, 200],
            'ensembleestimator__resiest__max_features':
                ['sqrt', 'log2', 20, 50],
        },
        # scoring=make_scorer(r2_score),
        n_jobs=8,
        cv=5,
        verbose=2,
    )
    g.fit(data, y)

    if save_checkpoint:
        joblib.dump(g, 'checkpoints/g.pkl')

    return g


def load_gridsearch(fpath='checkpoints/g.pkl'):
    if 'g' not in locals():
        g = joblib.load(fpath)
    return g


def summarize_fit(g, data, y):
    est = g.best_estimator_
    y_hat = est.predict(data)

    rmse = mean_squared_error(y, y_hat) ** 0.5
    mae = mean_absolute_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    best_params = g.best_params_

    pp = pprint.PrettyPrinter(indent=1)

    print('RMSE:')
    print(rmse)
    print()
    print('MAE:')
    print(mae)
    print()
    print('R-squared:')
    print(r2)
    print()
    print('Best parameters:')
    pp.pprint(g.best_params_)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # Load restaurant data
    data = fs.filter_yelp_data(fs.load_yelp_data())

    # Create revenue proxy
    y = fs.compute_revenue_proxy(data)
    assert len(data) == len(y)
    assert sum(np.isnan(y)) == 0

    print('''
    To tune hyperparameters:
        g = tune_estimator(data, y)

    To load previous search:
        g = joblib.load('checkpoints/g-1-ensemble.pkl')

    To summarize fit:
        summarize_fit(g, data, y)
    ''')
