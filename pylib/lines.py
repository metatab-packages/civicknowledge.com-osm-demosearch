"""

"""

from itertools import chain
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from demosearch import FileCache
from demosearch.util import run_mp
from shapely.wkt import loads as loads_wkt
from tqdm.notebook import tqdm
import appnope

tqdm.pandas()

import logging

lines_logger = logging.getLogger(__name__)

class LPError(Exception):
    pass


hw_type = {
    'residential': 'r',
    'primary': '1',
    'secondary': '2',
    'tertiary': '3',
    'motorway': 'm',
    'motorway_link ': 'l',
    'trunk': 't'
}


def get_cache(pkg):
    return FileCache(Path(pkg.path).parent.joinpath('data', 'cache'))


# Process each of the separate files, then
# write them back out for later recombination
def open_cache(pkg):
    pkg_root = Path(pkg.path).parent

    cache = FileCache(pkg_root.joinpath('data', 'cache'))

    if not cache.exists('hashes'):
        hashes = pkg.reference('us_geohashes').geoframe()
        cache.put_df('hashes', hashes)

    if not cache.exists('utm_grid'):
        utm_grid = pkg.reference('utm_grid').geoframe()
        cache.put_df('utm_grid', utm_grid)

    return cache


#
# Write out the lines files into chunks so we can run it in multiple
# processes

def estimate_lines(fp):
    """Estimate the number of lines in a very long line-oriented file"""
    lengths = []
    means = []
    sz = Path(fp).stat().st_size
    mean = 1
    std = 1
    ln = 1
    tq = tqdm(total=6000)  # SHould take less than 6K line to get estimate
    with fp.open() as f:

        while True:
            l = f.readline()

            if not l or (len(l) > 1000 and std < 2):
                return int(sz / mean)

            lengths.append(len(l))
            mean = np.mean(lengths).round(0)
            means.append(mean)
            std = np.std(means[-500:]).round(0)

            tq.update(1)
            tq.set_description(f"Est #lines {int(sz / mean)}")
            ln += 1


def split_lines(pkg, limit=None):
    cache = get_cache(pkg)

    try:
        # Returned the cached keys if this is already done
        return cache.get('splits/splits_keys')
    except KeyError:
        pass

    fp = pkg.reference('lines').resolved_url.fspath

    try:
        approx_lines = cache.config['lines_file_size']
    except KeyError:
        approx_lines = estimate_lines(fp)
        cache.config['lines_file_size'] = approx_lines

    chunksize = 10000
    total = int(approx_lines / chunksize)

    splits = []

    with pd.read_csv(fp, chunksize=chunksize, low_memory=False) as reader:
        for i, df in tqdm(enumerate(reader), total=total, desc='Split file'):
            if limit and i > limit:
                break
            key = f'splits/{i}'
            if not cache.exists(key):
                cache.put_df(key, df)
            splits.append(key)

    cache.put('splits/splits_keys', splits)

    return splits


def ro_key(rec_key):
    return f"recombine/{Path(rec_key).name}"


def f_run_overlay(cache_dir, key, okey):
    cache = FileCache(cache_dir)

    if cache.exists(okey):
        return okey

    t = cache.get_df(key)

    utm = cache.get_df('utm_grid')

    t = t[t.highway.isin(list(hw_type.keys()))]
    t['highway'] = t.highway.replace(hw_type)  # Cuts file size by 100M
    t['geometry'] = t.geometry.apply(shapely.wkt.loads)

    if len(t) == 0:
        return None

    gdf = gpd.GeoDataFrame(t, crs=4326)
    try:
        t = gpd.overlay(gdf, utm)

        try:
            cache.put_df(okey, t)
        except:
            if cache.exists(okey):
                cache.delete(key)
                raise
    except IndexError as e:

        raise LPError(f"Failed for {key} gdf:{len(gdf)} hashes:{len(utm)}: {e}")
    return okey


def run_overlay(pkg, splits, force=False):
    cache = get_cache(pkg)

    if not force:
        try:
            # Returned the cached keys if this is already done
            recombine_keys = cache.get('recombine/recombine_keys')

            if len(recombine_keys) == len(splits):
                return recombine_keys

        except KeyError:
            pass

    tasks = [[cache.root, e, ro_key(e)] for e in splits]

    recombine_keys = run_mp(f_run_overlay, tasks, desc='Overlay Geohash')

    cache.put('recombine/recombine_keys', recombine_keys)

    return list(filter(bool, recombine_keys))


def f_simplify_lines(cache_dir, key):
    cache = FileCache(cache_dir)

    if not key:
        return []

    try:
        df = cache.get_df(key)
    except EOFError as e:
        raise LPError(f"Failed to load key {key}: {e}")
    except AttributeError as e:
        raise LPError(f"Failed to load key {key}: {e}")

    okeys = []

    for idx, g in df.groupby('epsg'):
        _, fn = key.split('/')
        okey = f'simplified/{idx}/{fn}'

        if not cache.exists(okey):
            geometry = g.to_crs(epsg=idx).geometry \
                .simplify(20, False) \
                .to_crs(4326) \
                .apply(lambda e: shapely.wkt.dumps(e, rounding_precision=0))
            g = pd.DataFrame(g).assign(geometry=geometry)

            cache.put_df(okey, g)

        okeys.append(okey)

    return okeys


def simplify_lines(pkg, recombine_keys):
    cache = get_cache(pkg)

    try:
        # Returned the cached keys if this is already done
        return cache.get('simplified/simplified_keys')
    except KeyError:
        pass

    simplified_keys = run_mp(f_simplify_lines, [(cache.root, e) for e in recombine_keys],
                             desc='Simplify')

    simplified_keys = list(chain(*simplified_keys))

    cache.put('simplified/simplified_keys', simplified_keys)

    return simplified_keys


def write_files(pkg, simplified_keys):
    pkg_root = Path(pkg.path).parent
    cache = FileCache(pkg_root.joinpath('data', 'cache'))

    t = pd.concat([cache.get_df(e) for e in simplified_keys])
    t = t[['zone', 'epsg', 'us_state','cus_state', 'highway', 'geometry']]
    residential_roads = t[t.highway == 'r']
    nonres_roads = t[t.highway != 'r']

    residential_roads.to_csv(pkg_root.joinpath('data', 'residential_roads.csv'), index=False)
    nonres_roads.to_csv(pkg_root.joinpath('data', 'nonres_roads.csv'), index=False)

def build_lines(pkg):

    cache = open_cache(pkg)

    with appnope.nope_scope(): # Turn off AppNap on Macs
        lines_logger.info('Split the input file')
        splits = split_lines(pkg)
        lines_logger.info(f'   {len(splits)} splits keys')

        lines_logger.info('Run the overlay process')
        recombine_keys = run_overlay(pkg, splits, cache)
        print(f'   {len(recombine_keys)} recombine keys')

        if False:
            lines_logger.info('Simplify lines')
            simplified_keys = simplify_lines(pkg, recombine_keys)
            lines_logger.info(f'   {len(simplified_keys)} simplified keys')
        else:
            simplified_keys = recombine_keys

        lines_logger.info('Write the roads files')
        write_files(pkg, simplified_keys)
