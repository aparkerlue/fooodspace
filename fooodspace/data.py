from bokeh.palettes import d3
import numpy as np
import pandas as pd


def yelp_businesses_to_dataframe(data, y, exclude_unknown_categories=True):
    categories = [
        'Barbeque',
        'Spanish',
        'Middle Eastern',
        'Korean',
        'Latin American',
        'Asian Fusion',
        'Thai',
        'Indian',
        'French',
        'Mediterranean',
        'Chinese',
        'Mexican',
        'Japanese',
        'American (Traditional)',
        'American (New)',
        'Italian',
        # 'Bagels',
        # 'Bakeries',
        # 'Delis',
        # 'Fast Food',
        # 'Seafood',
        # 'Cafes',
        # 'Burgers',
        # 'Salad',
        # 'Pizza',
        # 'Sandwiches',
        # 'Breakfast & Brunch',
        # 'Coffee & Tea',
    ]
    colormap = {
        k: v
        for k, v in zip(
            reversed(categories),
            d3['Category20b'][len(categories)]
        )
    }
    L = []
    for d, r in zip(data, y):
        c = 'Other'
        for p in d['categories']:
            for q in categories:
                if p['title'] == q:
                    c = q
                    break
            else:
                continue
            break
        if exclude_unknown_categories and c == 'Other':
            continue
        L.append((
            d['id'],
            d['coordinates']['latitude'],
            d['coordinates']['longitude'],
            c,
            r,
            colormap[c],
        ))
    df = pd.DataFrame.from_records(
        L,
        columns=['id', 'lat', 'lon', 'category', 'revenue_proxy', 'catcolor'],
    )
    df_ = df.assign(
        revsize=np.round(1.5 * np.log(df['revenue_proxy'].values)).astype(int),
    )
    return df_


def count_restaurants_by_category(data, categories):
    """Count restaurants by category.

    `categories` is a data frame with columns `alias` and `title`.
    """
    catcounts = categories.assign(n=0)
    for d in data:
        for c in d['categories']:
            if c['alias'] in catcounts['alias'].values:
                catcounts.loc[catcounts['alias'] == c['alias'], 'n'] += 1
    return catcounts
