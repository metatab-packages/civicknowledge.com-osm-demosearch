# Get and process all of the state block files

import fiona
from pathlib import Path
import numpy as np
import metapack as mp
import geopandas as gpd
import pandas as pd
from auto_tqdm import tqdm
import rowgenerators as rg


from geoid.tiger import Cbsa
from demosearch.util import run_mp
from geoid.census import Block
from geoid.censusnames import stusab
from itertools import product
from shapely.wkt import loads

from .points import points_logger
from .util import get_cache


def _f_get_split_blocks(st, cache, url):
    k = f'blocks/geo/{st}'

    if not cache.exists(k):
        df = rg.geoframe(url).to_crs(4326)

        df['geoid'] = df.geoid20.apply(lambda v: str(Block.parse(v).as_acs()))
        df = df.rename(columns={
            'aland20': 'aland',
            'awater20': 'awater',
            'intptlat20': 'lat',
            'intptlon20': 'lon',
        })

        df = df[['geoid', 'aland', 'awater', 'lat', 'lon', 'geometry']]
        df['lat'] = df.lat.astype(float)
        df['lon'] = df.lon.astype(float)

        cache.put(k, df)

    return k


def split_blocks(pkg):
    """Download block files and cache them"""
    cache = get_cache(pkg)

    states = list(stusab.values())

    keys = run_mp(_f_get_split_blocks, [(st, cache, pkg.reference('block_templ').url.format(st=st))
                                        for st in states], n_cpu=4)  # 4 cpu b/c we're downloading

    return keys


def split_hashed_points(pkg):
    '''Split up the geohasned tags file'''

    cache = get_cache(pkg)

    ght = pkg.resource('point_tags').dataframe()

    break_start = [int(e) for e in np.linspace(0, len(ght) + 1, 10)]
    breaks = list(zip(break_start[:-1], break_start[1:]))
    parts = []

    for a, b in tqdm(breaks):
        k = f'blocks/hashed_points/{a}'
        cache.put_df(k, ght.loc[a:b - 1])

    return break_start


def _f_join_blocks(point_start, state, cache):
    k = f"blocks/joins/{point_start}-{state}"
    if cache.exists(k):
        return k

    try:
        pts = cache.get_df(f'blocks/hashed_points/{point_start}')
        pts['geometry'] = pts.geometry.apply(loads)
        pts = gpd.GeoDataFrame(pts, crs=4326)

        blks = cache.get(f'blocks/geo/{state}')

        j = gpd.sjoin(blks, pts)

        cache.put(k, j)

        return k
    except KeyError as e:
        return Exception("Failed for: " + k)


def join_blocks(pkg, break_starts):
    """Join census blocks and OSM points"""

    cache = get_cache(pkg)

    states = list(stusab.values())

    tasks = list(e + (cache,) for e in product(break_starts, states))

    keys = run_mp(_f_join_blocks, tasks)

    joins = [e for e in keys if not isinstance(e, Exception)]
    exn = [e for e in keys if isinstance(e, Exception)]

    return joins


def concat_osm_blocks(pkg, joins):
    cache = get_cache(pkg)

    parts = [cache.get(e) for e in joins]
    df = pd.concat(parts)

    cols = list(df.loc[:, 'amenity':].columns)
    block_osm = df[['geoid'] + cols].groupby('geoid').sum().reset_index()
    block_geo = df.loc[:, :'geometry'].drop_duplicates(subset=['geoid'])

    pkg_root = Path(pkg.path).parent
    block_osm.to_csv(pkg_root.joinpath('data', 'block_osm.csv'), index=False)
    block_geo.to_csv(pkg_root.joinpath('data', 'block_geo.csv'), index=False)

    return block_osm, block_geo


def build_osm_points(pkg):
    points_logger.debug("Spit blocks")
    split_blocks(pkg)

    points_logger.debug("Spit hashed points")
    break_starts = split_hashed_points(pkg)

    points_logger.debug("Join blocks")
    joins = join_blocks(pkg, break_starts)

    points_logger.debug("Concat points")
    return concat_osm_blocks(pkg, joins)


def _f_block_maps(cache, st, grid_key, cbsa_key):
    k1 = f'blocks/map/splits/cbsa/{st}'
    k2 = f'blocks/map/splits/utm/{st}'

    if not cache.exists(k1):
        df = cache.get(f'blocks/geo/{st}')

        cbsa = cache.get(cbsa_key)
        t = gpd.sjoin(df, cbsa)
        block_cbsa_map = t[['geoid_right', 'geoid_left']].rename(columns={'geoid_left': 'block', 'geoid_right': 'cbsa'})
        cache.put(k1, block_cbsa_map)

        utmg = cache.get(grid_key)
        t = gpd.sjoin(df, utmg)
        block_utm_map = t[['geoid', 'band', 'zone', 'epsg', 'cus_state']]
        cache.put(k2, block_utm_map)

    return k1, k2


def build_block_maps(pkg):
    cache = get_cache(pkg)
    states = list(stusab.values())

    grid_key = 'blocks/map/source/utm'
    cache.put(grid_key, pkg.reference('utm_grid').geoframe())

    cbsa_key = 'blocks/map/source/cbsa'
    cache.put(cbsa_key, pkg.reference('cbsa').geoframe().to_crs(4326))

    tasks = [(cache, st, grid_key, cbsa_key) for st in states]

    try:
        import appnope
        with appnope.nope_scope():
            r = run_mp(_f_block_maps, tasks)
    except ImportError:
        r = run_mp(_f_block_maps, tasks)

    cbsa_map = pd.concat([cache.get(e[0]) for e in r])\

    utm_map = pd.concat([cache.get(e[1]) for e in r])

    pkg_root = Path(pkg.path).parent
    cbsa_map.to_csv(pkg_root.joinpath('data', 'cbsa_map.csv'), index=False)
    utm_map.to_csv(pkg_root.joinpath('data', 'utm_map.csv'), index=False)