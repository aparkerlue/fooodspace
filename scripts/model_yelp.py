import gzip
import logging
import os
import re

import ujson as json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_union, make_pipeline


def is_iterable(x):
    """Return True if an object is iterable."""
    try:
        iter(x)
    except TypeError:
        return False
    return True


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, colnames, pd_out=False):
        self.colnames = colnames
        self.pd_out = pd_out   # Produce a pandas DataFrame

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Return selected columns."""
        X_ = [
            tuple(self._extract_column(d, c) for c in self.colnames)
            for d in X
        ]
        if self.pd_out:
            return pd.DataFrame.from_records(X_, columns=self.colnames)
        else:
            return X_

    def _extract_column(self, record, colname):
        """Extract a column from a record."""
        if colname in ['latitude', 'longitude']:
            x = record['coordinates'][colname]
            return x if x is not None else np.nan
        elif colname == 'categories':
            L = sorted(set(d['title'] for d in record[colname]))
            if self.pd_out:
                return ','.join(L) if len(L) > 0 else np.nan
            else:
                return L
        elif colname == 'transactions':
            L = sorted(set(t for t in record[colname]))
            if self.pd_out:
                return ','.join(L) if len(L) > 0 else np.nan
            else:
                return L
        elif colname == 'price':
            try:
                return len(record[colname])
            except KeyError:
                return np.nan
        elif colname == 'display_address':
            x = record['location'][colname]
            return ', '.join(x)
        else:
            try:
                return record[colname]
            except KeyError:
                return np.nan


class DictEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform list of tuples into dict of 1's."""
        X_ = []
        for t in X:
            d = {}
            for L in t:
                d = {**d, **{k: 1 for k in L}}
            X_.append(d)
        return X_


class AvenueParseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform list of tuples of addresses to ave indicators."""
        X_ = []
        for x in X:
            if len(x) == 0:
                x_ = np.nan
            elif len(x) == 1:
                x_ = 0 if re.search('(Ave|Broadway)', x[0]) is None else 1
            else:
                raise ValueError('Received list of non-unit-length tuples')
            X_.append((x_, ))
        return X_


def filter_businesses_in_categories(data, categories):
    data_ = [
        d
        for d in data
        if any(c['alias'] in categories for c in d['categories'])
    ]
    return data_


def extract_yelp_categories(data):
    """Extract unique categories from list of Yelp business dicts."""
    return pd.DataFrame.from_records(
        sorted(set(
            (c['alias'], c['title'])
            for x in data
            for c in x['categories']
        )),
        columns=['alias', 'title'],
    )


def compute_revenue_proxy(data):
    yelp_prices = pd.read_csv('data/yelp_prices.csv')

    y = []
    for _, r in (
        ColumnSelector(['price', 'review_count'], pd_out=True)
        .fit_transform(data)
        .iterrows()
    ):
        price_symbol = (
            int(r.at['price']) * '$'
            if not np.isnan(r.at['price'])
            else None
        )
        price_avg = (
            yelp_prices.set_index('price_range')
            .at[price_symbol, 'price_avg']
        ) if price_symbol is not None else np.nan
        review_count = r.at['review_count']
        y.append(price_avg * review_count)
    return y


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # Load restaurant data
    fpath = os.path.join(
        'data',
        'yelp_business_lookup_manhattan_restaurants-20180203.json.gz',
    )
    with gzip.open(fpath, 'rt') as f:
        data = [json.loads(line) for line in f]
    logging.info('Loaded {:,} business records'.format(len(data)))

    # Check that all restaurants are open
    assert len([x['id'] for x in data if x['is_closed']]) == 0

    # Count restaurants filtered through categories
    yc = pd.read_csv('data/yelp_categories_filter.csv')
    data = filter_businesses_in_categories(data, yc['alias'].values)
    n = len(data)
    assert n == 10844
    logging.info(
        'Filtered {:,} restaurants with target categories'
        .format(n)
    )

    # Check that category titles don't include commas
    assert not any([',' in s for s in yc['title'].values])

    # Find unique transaction values
    assert (
        set(t for d in data for t in d['transactions'])
        == {'delivery', 'pickup', 'restaurant_reservation'}
    )

    # Create feature set
    tfr_categories = make_pipeline(
        ColumnSelector(['categories']),
        DictEncoder(),
        DictVectorizer(),
    )
    tfr_avenues = make_pipeline(
        ColumnSelector(['display_address']),
        AvenueParseTransformer(),
    )
    tfr_transactions = make_pipeline(
        ColumnSelector(['transactions']),
        DictEncoder(),
        DictVectorizer(),
    )
    featunion = make_union(
        ColumnSelector([
            'latitude',
            'longitude',
            'price',
            'rating',
        ]),
        tfr_avenues,
        tfr_categories,
        tfr_transactions,
    )
    X = featunion.fit_transform(data)

    # Remove observations with missing data
    i_data = np.flatnonzero(~np.isnan(X.toarray()).any(axis=1))
    logging.info(
        'Found {:,} restaurants with full feature set'
        .format(len(i_data))
    )
    X = X[i_data]

    # Create dependent variable
    logging.info('Computing revenue proxy...')
    y = compute_revenue_proxy([data[i] for i in i_data])

    # Features to add:
    # - Number of words in name
    # - Hours

    # TODO: Check if we have redundantcategory values
