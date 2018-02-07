from datetime import datetime, time
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
        elif colname == 'hours_start':
            # Find earliest opening and latest closing times across
            # all days of the week.
            try:
                x = record['hours']
            except KeyError:
                hours = None
            else:
                hours = self._yelp_hours_get_min_start(x[0]['open'])
            return np.nan if self.pd_out and hours is None else hours
        elif colname == 'hours_end':
            try:
                x = record['hours']
            except KeyError:
                hours = None
            else:
                hours = self._yelp_hours_get_max_end(x[0]['open'])
            return np.nan if self.pd_out and hours is None else hours
        else:
            try:
                return record[colname]
            except KeyError:
                return np.nan

    @staticmethod
    def _yelp_hours_get_min_start(L_open):
        """Find minimum start time from Yelp's `open' list.

        The `open' list is inside the `hours' field of a business
        entry.

        `day' 0 is Monday, and `day' 6 is Sunday.

        Here's an example:

        'hours': [{'hours_type': 'REGULAR',
          'is_open_now': True,
          'open': [{'day': 0, 'end': '2300', 'is_overnight': False, 'start': '0800'},
           {'day': 1, 'end': '2300', 'is_overnight': False, 'start': '0800'},
           {'day': 2, 'end': '2300', 'is_overnight': False, 'start': '0800'},
           {'day': 3, 'end': '2300', 'is_overnight': False, 'start': '0800'},
           {'day': 4, 'end': '2300', 'is_overnight': False, 'start': '0800'},
           {'day': 5, 'end': '2300', 'is_overnight': False, 'start': '0900'},
           {'day': 6, 'end': '2200', 'is_overnight': False, 'start': '0900'}]}]
        """
        starts = []
        for d in L_open:
            try:
                start = d['start']
            except KeyError:
                continue
            x = datetime.strptime(start, '%H%M').time()
            starts.append(x)
        return min(starts).strftime('%H:%M')

    @staticmethod
    def _yelp_hours_get_max_end(L_open):
        """Find maximum end time from Yelp's `open' list."""
        ends = []
        for d in L_open:
            try:
                end = d['end']
            except KeyError:
                continue
            x = datetime.strptime(end, '%H%M').time()
            ends.append(x)
        return max(ends).strftime('%H:%M')


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


def time_diff_abs(x, y):
    """Return the absolute difference between two time objects."""
    x, y = sorted([x, y], reverse=True)
    d = x.hour - y.hour + (x.minute - y.minute) / 60
    return time(int(d), round((d - int(d)) * 60))


class HourStartTransformer(BaseEstimator, TransformerMixin):
    """Transform start times into hours before noon."""

    MAX_START = time(12, 0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = []
        for x in X:
            t = x[0]
            if t is None:
                x_ = np.nan
            else:
                t_ = datetime.strptime(t, '%H:%M').time()
                if t_ < self.MAX_START:
                    dt = time_diff_abs(t_, self.MAX_START)
                    x_ = dt.hour + dt.minute / 60
                else:
                    x_ = 0.0
            X_.append((x_, ))
        return X_


class HourEndTransformer(BaseEstimator, TransformerMixin):
    """Transform end times into hours after 19:00."""

    MIN_END = time(19, 0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = []
        for x in X:
            t = x[0]
            if t is None:
                x_ = np.nan
            else:
                t_ = datetime.strptime(t, '%H:%M').time()
                if t_ > self.MIN_END:
                    dt = time_diff_abs(t_, self.MIN_END)
                    x_ = dt.hour + dt.minute / 60
                else:
                    x_ = 0.0
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
    return np.asarray(y, dtype=float)


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
    tfr_hours_start = make_pipeline(
        ColumnSelector(['hours_start']),
        HourStartTransformer(),
    )
    tfr_hours_end = make_pipeline(
        ColumnSelector(['hours_end']),
        HourEndTransformer(),
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
        tfr_hours_start,
        tfr_hours_end,
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
    assert sum(np.isnan(y)) == 0

    # TODO: Check if we have redundant category values
