import logging

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

import fooodspace as fs

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # Load restaurant data
    data = fs.filter_yelp_data(fs.load_yelp_data())

    # Create revenue proxy
    y = fs.compute_revenue_proxy(data)
    assert len(data) == len(y)
    assert sum(np.isnan(y)) == 0

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
            # 'ensembleestimator__baseest__alpha': [.001, .003, .01, .03, .1, .3],
            'ensembleestimator__baseest__alpha': [.02, .025, .03, .035, .04],
            # KNeighborsRegressor:
            # 'ensembleestimator__resiest__n_neighbors': range(4, 7),
            # RandomForestRegressor:
            # 'ensembleestimator__resiest__min_samples_leaf': [1, 5, 10, 15, 20, 25],
            'ensembleestimator__resiest__min_samples_leaf': [4, 5, 6, 7, 8],
        },
        # scoring=make_scorer(r2_score),
        n_jobs=4,
        cv=3,
        verbose=2,
    )
    g.fit(data, y)

    joblib.dump(g, 'checkpoints/g.pkl')

    if 'g' not in locals():
        g = joblib.load('checkpoints/g.pkl')

    print(g.best_params_)

    print(r2_score(y, g.predict(data)))  # 0.63

    print(np.sqrt(-g.score(data, y)))
