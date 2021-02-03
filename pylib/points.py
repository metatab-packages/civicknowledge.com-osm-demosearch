"""

"""
from itertools import chain

import geopandas as gpd
import libgeohash as gh
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely.wkt import loads as loads_wkt
from tqdm import tqdm
tqdm.pandas()
from pathlib import Path

from demosearch.util import run_mp
from .lines import open_cache

extract_tags = ['amenity', 'tourism', 'shop', 'leisure', 'natural', 'parking']


def _extract_tags(df, extract_tags):
    from sqlalchemy.dialects.postgresql import HSTORE

    h = HSTORE()
    f = h.result_processor(None, None)

    # Prune the dataset to just the records that have the tags we want.
    # before getting to the more expensive operation of extracting the tags.
    # This should reduce the dataset from 24M rows to less than 6M.
    t = df.dropna(subset=['other_tags'])
    t = t[t.highway.isnull()]

    flags = [t.other_tags.str.contains(e) for e in extract_tags]
    comb_flags = [any(e) for e in list(zip(*flags))]

    t = t[comb_flags]

    rows = []
    errors = []
    for idx, r in t.set_index('osm_id')[['other_tags']].iterrows():
        try:
            d = f(r.other_tags)
            rows.append([idx] + [d.get(e) for e in extract_tags])
        except TypeError as e:
            errors.append(r, e)

    return (rows, errors)


def make_tags_df(pkg):
    """Create the tags dataframe"""
    cache = open_cache(pkg)

    try:
        tags_df = cache.get_df('points/tags_df')
    except KeyError:
        points_df = pkg.reference('points').read_csv(low_memory=False)

        # Split the file and extract tags in multiprocessing
        N_task = 200
        tasks = [(e, extract_tags) for e in np.array_split(points_df, N_task)]

        results = run_mp(_extract_tags, tasks, 'Split OSM other_tags')
        tags = list(chain(*[e[0] for e in results]))
        errors = list(chain(*[e[1] for e in results]))

        tags_df = pd.DataFrame(tags, columns=['osm_id'] + extract_tags)

        # 1/2 the entries, 2.7M are trees and rocks
        tags_df = tags_df[~tags_df.natural.isin(['tree', 'rock'])]

        tags_df = pd.merge(tags_df, points_df[['osm_id', 'geometry']], on='osm_id')

        def encode(v):
            return gh.encode(*list(map(float, v[7:-1].split()))[::-1])

        tags_df['geohash'] = tags_df.geometry.progress_apply(encode)

        tags_df['geometry'] = tags_df.geometry.progress_apply(shapely.wkt.loads)

        tags_df = gpd.GeoDataFrame(tags_df, geometry='geometry', crs=4326)

        cache.put_df('points/tags_df', tags_df)

    return tags_df





def extract_class_columns(tags_df):
    tags_df['class'] = tags_df.loc[:, ('amenity', 'tourism', 'shop', 'leisure', 'natural', 'parking')].fillna(
        method='ffill', axis=1).fillna(method='bfill', axis=1).iloc[:, 0]

    replace = {'parking': 'parking_space',
               'pub': 'bar',
               }
    cls = ['restaurant', 'bar', 'cafe', 'fast_food', 'supermarket', 'grave_yard', 'playground',
           'bicycle_parking', 'park', 'fuel', 'bank', 'hotel', 'fitness_centre',
           'laundry', 'clothes', 'convenience', 'parking', 'parking_space']

    t = tags_df[['geohash', 'class']].replace(replace)
    t = t[t['class'].isin(cls)]

    cls_df = t.groupby([t.geohash.str.slice(0, 8), 'class']).count().unstack().fillna(0).droplevel(0, axis=1)

    return cls_df


def make_geotags_df(pkg, tags_df, cls_df):
    # At 8 digits, geohashes are, on average 4m by 20M over the US
    # At 6, 146m x 610m
    # At 4, 4Km x 20Km
    # Clip to 5 because it's really unlikely that there are actually more than 10
    # amenities in a cell.

    pkg_root = Path(pkg.path).parent

    group_counts = tags_df.groupby(tags_df.geohash.str.slice(0, 8)) \
        [['amenity', 'tourism', 'shop', 'leisure', 'natural', 'parking']].count().clip(0, 10)

    t = group_counts.join(cls_df, how='outer').fillna(0).astype(int)

    t['geometry'] = [Point(gh.decode(e)[::-1]) for e in t.index]

    geohash_tags = gpd.GeoDataFrame(t, geometry='geometry', crs=4326).reset_index()

    geohash_tags.to_csv(pkg_root.joinpath('data', 'residential_roads.csv'))

    return geohash_tags