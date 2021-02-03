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

tqdm.pandas()


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


# Process each of the separate files, then
# write them back out for later recombination
def open_cache(pkg):
    pkg_root = Path(pkg.path).parent

    cache = FileCache(pkg_root.joinpath('data', 'cache'))

    if not cache.exists('hashes'):
        hashes = pkg.reference('us_geohashes').geoframe()
        cache.put_df('hashes', hashes)

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


def split_data(pkg, cache, limit=None):
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
    approx_lines = 53065618  # via wc
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

    hashes = cache.get_df('hashes')

    t = t[t.highway.isin(list(hw_type.keys()))]
    t['highway'] = t.highway.replace(hw_type)  # Cuts file size by 100M
    t['geometry'] = t.geometry.apply(shapely.wkt.loads)

    if len(t) == 0:
        return None

    gdf = gpd.GeoDataFrame(t, crs=4326)
    try:
        t = gpd.overlay(gdf, hashes)
        t = t[['osm_id', 'geohash', 'utm_epsg', 'utm_area', 'highway', 'geometry']]
        try:
            cache.put_df(okey, t)
        except:
            if cache.exists(okey):
                cache.delete(key)
                raise
    except IndexError as e:

        raise LPError(f"Failed for {key} gdf:{len(gdf)} hashes:{len(hashes)}: {e}")
    return okey


def run_overlay(splits, cache, force=False):
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

    for idx, g in df.groupby('utm_epsg'):
        _, fn = key.split('/')
        okey = f'simplified/{idx}/{fn}'

        if not cache.exists(okey):
            geometry = g.to_crs(epsg=idx).geometry \
                .simplify(20, False) \
                .apply(lambda e: shapely.wkt.dumps(e, rounding_precision=0))
            g = pd.DataFrame(g).assign(geometry=geometry)

            cache.put_df(okey, g)

        okeys.append(okey)

    return okeys

def simplify_lines(cache, recombine_keys):
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
    t = t[['geohash', 'highway', 'geometry']]
    residential_roads = t[t.highway == 'r']
    nonres_roads = t[t.highway != 'r']

    residential_roads.to_csv(pkg_root.joinpath('data', 'residential_roads.csv'))
    nonres_roads.to_csv(pkg_root.joinpath('data', 'nonres_roads.csv'))


def geohash_aggregate(recombine_keys):
    """Break lines into segments, assign them to 7 digit geohashes by the location
    of one end, and aggreate"""

    def _f(key):
        df = cache.get_df(key)
        okeys = []
        errs = []
        rows = []
        for idx, g in df.groupby('utm_epsg'):
            _, fn = key.split('/')
            okey = f'lengths/{idx}/{fn}'

            tr = Transformer.from_crs(idx, 4326)  # Transform back to lat/lon for geohash encoding

            for gidx, r in g.to_crs(epsg=idx).iterrows():
                try:
                    a = np.array(r.geometry)

                    for l, p in zip(np.linalg.norm(a[:-1] - a[1:], 2, 1), a[:-1]):  # compute the length of each segment
                        lon, lat = tr.transform(*p)
                        geohash = gh.encode(lat, lon, 7)
                        rows.append([geohash, r.highway, int(l)])
                except TypeError:
                    pass

            t = pd.DataFrame(rows, columns='geohash road_type len'.split())
            t = t.groupby(['geohash', 'road_type']).sum().reset_index()

            cache.put_df(okey, t)
            okeys.append(okey)

        return okeys

    epsg_keys = run_mp(_f, [(e,) for e in recombine_keys], desc='Split By Geohash')

    t = pd.concat([cache.get_df(e) for e in list(chain(*epsg_keys))])
    t = t.groupby(['geohash', 'road_type']).sum().reset_index()
    residential_hash = t[t.road_type == 'r']
    nonres_hash = t[t.road_type != 'r']
