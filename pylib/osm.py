from itertools import chain

import geopandas as gpd
import libgeohash as gh
import metapack as mp
import numpy as np
import pandas as pd
import shapely
from auto_tqdm import tqdm
from demosearch.install import logger
#
from demosearch.util import gh_data_path, munge_pbar, run_mp
from shapely.geometry import Point
from tqdm import tqdm

class OsmProcessor(object):

    def __init__(self, cache, progress_bar=None):

        self.cache = cache
        self.pbar = munge_pbar(progress_bar)

        self.osm_key = 'osm'
        self.csv_key = 'osm/csv'

        self.extract_tags = ['amenity', 'tourism', 'shop', 'leisure', 'natural', 'parking']

        self.tags = None
        self.errors = None

    def convert_pbf(self, input_fn, output_dir=None):
        """Convert the an OSM PBF file to CSV so we can do something with it. """
        from subprocess import Popen, PIPE

        lco = '-lco GEOMETRY=AS_WKT -lco GEOMETRY_NAME=geometry -lco CREATE_CSVT=yes'

        if not output_dir:
            output_dir = self.cache.joinpath(self.csv_key)

        if output_dir.exists():
            raise Exception("Output directory should not exist")

        # output_dir.mkdir(parents=True)

        cmd = f'ogr2ogr  -f "CSV" -skipfailures {lco} {output_dir} {input_fn}'

        print("Running: ", cmd)

        process = Popen(cmd, stdout=PIPE, shell=True)

        while True:
            line = process.stdout.read()
            if not line:
                break
            print(line)



    def osm_points(self, nrows=None):
        """Load the OSm points file"""
        p = self.cache.joinpath(self.csv_key, 'points.csv')
        return pd.read_csv(p, low_memory=False, nrows=nrows)


    def get_tags_df(self, nrows=None, force=False):
        """Extract all of the OSM points records that have useful other_tags,
        add geohashes, and write the file to the cache"""

        tqdm.pandas() # Add progress_apply()

        key = self.osm_key + '/tags'

        if self.cache.exists(key) and not force:
            return self.cache.get_df(key)

        p = self.cache.joinpath(self.csv_key, 'points.csv')

        logger.debug('Loading points file')
        df = pd.read_csv(p, low_memory=False, nrows=nrows)

        logger.debug('Generate tasks')
        tasks = [(e, self.extract_tags) for e in np.array_split(df, 200)]

        results = run_mp(OsmProcessor.do_extract_tags, tasks, 'Split OSM other_tags')
        self.tags = list(chain(*[e[0] for e in results]))
        self.errors = list(chain(*[e[1] for e in results]))

        logger.debug('Create tags df')
        tags_df = pd.DataFrame(self.tags, columns=['osm_id'] + self.extract_tags)

        # 1/2 the entries, 2.7M are trees and rocks
        logger.debug('Remove trees and rock')
        tags_df = tags_df[~tags_df.natural.isin(['tree', 'rock'])]

        logger.debug('Merge geometry')
        tags_df = pd.merge(tags_df, df[['osm_id', 'geometry']], on='osm_id')

        def encode(v):
            return gh.encode(*list(map(float, v[7:-1].split()))[::-1])

        logger.debug('Add geohash')
        tags_df['geohash'] = tags_df.geometry.progress_apply(encode)

        logger.debug('Convert to geopandas')
        tags_df['geometry'] = tags_df.geometry.progress_apply(shapely.wkt.loads)

        tags_df = gpd.GeoDataFrame(tags_df, geometry='geometry', crs=4326)

        logger.debug('Write to file')
        self.cache.put_df(key, tags_df)

        return tags_df


    def get_tag_counts(self, force = False):
        """Coalesce the tags file counts per geohash, for 8-digit geohashes"""

        key = self.osm_key + '/tag_counts'

        if self.cache.exists(key) and not force:
            return self.cache.get_df(key)

        logger.debug('Get tags dataset')
        tags = self.get_tags_df()

        tags['class'] = tags.loc[:, ('amenity', 'tourism', 'shop', 'leisure', 'natural', 'parking')].fillna(
            method='ffill', axis=1).fillna(method='bfill', axis=1).iloc[:, 0]

        replace = {'parking': 'parking_space',
                   'pub': 'bar',
                   }
        cls = ['restaurant', 'bar', 'cafe', 'fast_food', 'supermarket', 'grave_yard', 'playground',
               'bicycle_parking', 'park', 'fuel', 'bank', 'hotel', 'fitness_centre',
               'laundry', 'clothes', 'convenience', 'parking', 'parking_space']

        t = tags[['geohash', 'class']].replace(replace)
        t = t[t['class'].isin(cls)]
        # t['dummy'] = 1

        logger.debug('Groupby classes geohash')
        cls_df = t.groupby([t.geohash.str.slice(0, 8), 'class']).count().unstack().fillna(0).droplevel(0, axis=1)

        cls_df.head()

        # At 8 digits, geohashes are, on average 4m by 20M over the US
        # At 6, 146m x 610m
        # At 4, 4Km x 20Km
        # Clip to 5 because it's really unlikely that there are actually more than 10
        # amenities in a cell.
        logger.debug('Groupby Tags geohash')
        group_counts = tags.groupby(tags.geohash.str.slice(0, 8))[
            ['amenity', 'tourism', 'shop', 'leisure', 'natural', 'parking']].count().clip(0, 10)
        group_counts.head()

        logger.debug('Join datasets')
        df = group_counts.join(cls_df, how='outer').fillna(0).astype(int)

        logger.debug('Adding geometry')
        df['geometry'] = [Point(gh.decode(e)[::-1]) for e in df.index]

        df = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)

        key = self.osm_key + '/tag_counts_4326'
        self.cache.put_df(key, df)

        logger.debug('Convert to UTM')
        df = df.to_crs(utm_crs).reset_index()

        key = self.osm_key + '/tag_counts'
        self.cache.put_df(key, df)

        return df

    def write_tagcounts_by_hash(self, force=False):

        tags = self.get_tag_counts().reset_index()

        for gh4, g_df in self.pbar(tags.groupby(tags.geohash.str.slice(0, 4)),desc='Write Tags Counts by Hash'):
            key = gh_data_path(gh4, 'tagcounts.pkl')
            if force or not self.cache.exists(key):
                self.cache.put_df(key, g_df)

    def get_tagcounts_by_hash(self, ghc):
        key = gh_data_path(ghc, 'tagcounts.pkl')
        return self.cache.get(key)

    def get_expanded_tagcounts_by_hash(self, ghc):
        """Like get_tracts_by_hash, but also get the 8 surrounding hash areas"""

        def _get(ghc):
            try:
                return self.get_tagcounts_by_hash(ghc)
            except KeyError:
                return None

        gh4e = gh.expand(ghc[0:4])
        frames = [_get(e) for e in gh4e]

        if not frames:
            raise KeyError

        t = pd.concat([e for e in frames if e is not None])

        return t