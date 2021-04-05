
import fiona
from pathlib import Path
import metapack as mp
import geopandas as gpd
import pandas as pd
import numpy as np
from auto_tqdm import tqdm

from demosearch.util import run_mp
from .util import get_cache

import logging

cluster_logger = logging.getLogger(__name__)


class ClustersException(Exception):
    pass

def get_points_tags(pkg, cache):

    key = '/clusters/point_tags'
    if not cache.exists(key):
        pt  = pkg.resource('point_tags').geoframe()
        cache.put(key, pt)
    else:
        pt = cache.get(key)

    return pt

def point_groups(df):
    """Reduce the set of tags down to a smaller set"""
    df = df.copy()

    groups = {
        'entertain': ['cafe', 'restaurant', 'bar'],
        'casual': ['fast_food', 'convenience'],
        'shop': ['shop', 'clothes', 'supermarket', 'bank', 'laundry', 'parking'],
        'active': ['playground', 'bicycle_parking', 'fitness_centre', 'park'],
        'travel': ['fuel', 'hotel', 'amenity', 'tourism', 'leisure', 'natural']
    }

    for agg, cols in groups.items():
        # Reduce all of the layers to 1 per geohash. Anymore than that is probably spurious
        df.loc[:, cols] = (df.loc[:, cols] > 0).astype(np.int8)
        df[agg] = df[cols].sum(axis=1)

    return df.loc[:, ['geoid', 'geohash', 'geometry'] + list(groups.keys())]


def link_elements(a_ids, b_ids):

    cluster_n  = 0
    clusters = {}

    def find_cluster(clusters, a ,b):
        if a in clusters:
            return clusters[a]
        if b in clusters:
            return clusters[b]
        return None


    for a, b in  zip(a_ids, b_ids):
        a = int(a)
        b = int(b)
        c = find_cluster(clusters, a ,b)

        if c is None:
            c  = cluster_n
            cluster_n += 1

        clusters[a] = c
        clusters[b] = c

    return clusters


def rebuild_geo(clusters, df):
    cdf = pd.DataFrame(clusters.items(), columns=['index', 'cluster_n']).set_index('index')

    t = df.join(cdf)
    if len(t) == 0:
        raise ClustersException(f'Empty dataframe {len(df)} {len(clusters)}')
    t.index.name = None # index gets names cluster_n, which conflicts with cluster_n column
    t = t.groupby('cluster_n').apply(lambda g: g.unary_union)

    g = gpd.GeoDataFrame({'geometry': t},crs=df.crs)
    return g

def merge_points(df):

    if len(df) == 0:
        raise ClustersException(f'Empty dataframe')

    t = gpd.sjoin(df, df, op='intersects')
    clusters = link_elements(t.index, t.index_right)

    if len(clusters) == 0 or len(t) == 0:
        raise ClustersException(f'Empty dataframe {len(df)}')

    return rebuild_geo(clusters, t)

def to_gdf(s, crs):
    return gpd.GeoDataFrame({'geometry': s}, crs=crs)

def rebuffer_points(points):
    from shapely.geometry import box
    t = points.buffer(150).bounds
    t = gpd.GeoSeries([box(*r.to_list()) for idx, r in t.iterrows()]).unary_union \
        .simplify(20).buffer(20)

    return gpd.GeoSeries(t, crs=points.crs)


def multi_buffer_and_merge(df):
    """Buffer and merge points multiple times to build clusters"""

    df = df.copy()

    df['geometry'] = df.buffer(60)

    g1 = merge_points(df)

    g = to_gdf(g1.buffer(30), df.crs)
    g2 = merge_points(g)

    g = to_gdf(g2.buffer(30), df.crs)
    return merge_points(g)




def get_lines(pkg, cache):
    key = '/clusters/lines'
    if not cache.exists(key):
        pt = pkg.resource('nonres_roads').geoframe()
        cache.put(key, pt)
    else:
        pt = cache.get(key)

    return pt


def cache_lines_cbsa(pkg, cache):

    if not cache.exists('/clusters/source_lines/cbsa/31000US41740'):  # San Diego

        cbsa = pkg.reference('cbsa').geoframe().to_crs(4326)
        pt1 = get_lines(pkg, cache)

        pt2 = gpd.sjoin(pt1, cbsa[['geometry', 'geoid']], how='left')
        pt2 = pt2.drop(columns=['index_right'])

        for idx, g in tqdm(pt2.groupby('geoid')):
            key = f'/clusters/source_lines/cbsa/{idx}'
            cache.put(key, g)

def cache_points_cbsa(pkg, cache):
    if not cache.exists('/clusters/source_points/cbsa/31000US41740'):  # San Diego
        utm_grid = pkg.reference('utm_grid').geoframe()
        pt = get_points_tags(pkg, cache)
        t = gpd.sjoin(pt, utm_grid)
        for idx, g in tqdm(t.groupby('geoid')):
            key = f'/clusters/source_points/cbsa/{idx}'
            cache.put(key, g)


def build_buffered_clusters(cache, geoid):
    k1 = f'/clusters/points/{geoid}'
    k2 = f'/clusters/buffered/{geoid}'

    if not cache.exists(k1):
        points = cache.get(f'/clusters/source_points/cbsa/{geoid}')
        lines = cache.get(f'/clusters/source_lines/cbsa/{geoid}')
        epsg = int(points.epsg.value_counts().index[0])
        mpg = point_groups(points).to_crs(epsg)  # .drop(columns=['index_right', 'index'])

        clusters = multi_buffer_and_merge(mpg)
        clusters = clusters.reset_index()
        clusters['cluster_n'] = clusters.index
        t = gpd.overlay(lines.to_crs(clusters.crs), clusters)
        t = t.groupby('cluster_n').apply(lambda g: g.unary_union)
        t = merge_points(to_gdf(to_gdf(t, epsg).buffer(100), epsg))
        buffered_clusters = t.reset_index()

        cache.put(k1, mpg.to_crs(4326).assign(cbsa=geoid))
        cache.put(k2, buffered_clusters.to_crs(4326).assign(cbsa=geoid))

    return k1, k2

def run_cbsa_clusters(cache, geoid):

    try:
        return build_buffered_clusters(cache, geoid)
    except ClustersException as e:
        return (e, geoid)
    except Exception as e:
        return (e, geoid)


def build_clusters(pkg):

    cache = get_cache(pkg)

    cluster_logger.info('Caching source points by CBSA')
    cache_points_cbsa(pkg, cache)
    cache_lines_cbsa(pkg, cache)

    cluster_logger.info('Start MP run')
    tasks = [(cache, e.stem) for e in cache.list('clusters/source_points/cbsa')]
    r = run_mp(run_cbsa_clusters, tasks)

    cluster_logger.info('Assemble metro points')
    metro_point_keys = [k1 for k1, k2 in r if not isinstance(k1, Exception)]
    frames = [cache.get(k) for k in tqdm(metro_point_keys)]
    metro_points = pd.concat(frames)

    cluster_logger.info('Assemble clusters')
    cluster_keys = [k2 for k1, k2 in r if not isinstance(k1, Exception)]
    frames = [cache.get(k) for k in tqdm(cluster_keys) if cache.exists(k)]
    clusters = pd.concat(frames)

    cluster_logger.info('Write files')
    pkg_root = Path(pkg.path).parent
    metro_points.to_csv(pkg_root.joinpath('data', 'metro_points.csv'), index=False)
    clusters.to_csv(pkg_root.joinpath('data', 'business_clusters.csv'), index=False)