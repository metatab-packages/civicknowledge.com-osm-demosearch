# Task definitions for invoke
# You must first install invoke, https://www.pyinvoke.org/



import sys
from pathlib import Path
import metapack as mp

from metapack.appurl import SearchUrl
SearchUrl.initialize()  # This makes the 'index:" urls work

sys.path.append(str(Path(__file__).parent.resolve()))

import pylib


pbf_file = 'north-america-latest.osm.pbf'
pbf_url = f'https://download.geofabrik.de/{pbf_file}'

# You can also create you own tasks
from invoke import task

from metapack_build.tasks.package import ns, build as mp_build

# To configure options for invoke functions you can:
# - Set values in the 'invoke' section of `~/.metapack.yaml
# - Use one of the other invoke config options:
#   http://docs.pyinvoke.org/en/stable/concepts/configuration.html#the-configuration-hierarchy
# - Set the configuration in this file:

# ns.configure(
#    {
#        'metapack':
#            {
#                's3_bucket': 'bucket_name',
#                'wp_site': 'wp sot name',
#                'groups': None,
#                'tags': None
#            }
#    }
# )

@task
def get_pbf(c):
    if not Path.cwd().joinpath('data',pbf_file).exists():
        c.run(f'curl --progress-bar -o data/{pbf_file} {pbf_url}')
    else:
        print(f'File data/{pbf_file} already exists')

ns.add_task(get_pbf)

@task
def convert_pbf(c):
    """Run the conversion of the OSM file"""
    import sys
    
    from demosearch.osm import OsmProcessor
    op = OsmProcessor(None)
    
    p = Path.cwd().joinpath('data',pbf_file)
    out_dir = p.parent.joinpath('csv')
    
    if out_dir.joinpath('lines.csv').exists():
        print("PBF file is already converted")
        return
    
    if not p.exists():
        print("ERROR: Download https://download.geofabrik.de/north-america-latest.osm.pbf and put it in the data directory"
              " or run `invoke get_pbf`")
        
        sys.exit(1)
    
    if out_dir.exists():
        print(f"ERROR: output dir {out_dir} should not exist")
        sys.exit(1)
        
   
    op.convert_pbf(p, out_dir)


ns.add_task(convert_pbf)

@task
def create_roads_files(c):
    """Build the residential_roads.csv and nonres_roads.csv files"""
    cache_dir = str(Path(__file__).parent.resolve())
    print(f"Cache: {cache_dir}")
    
    pkg = mp.open_package(cache_dir)
    
    cache = pylib.open_cache(pkg)

    print('-- Convert PBF file')
    convert_pbf(c)

    print('-- Split the input file')
    splits = pylib.split_data(pkg, cache)
    print(f'   {len(splits)} splits keys')
    
    print('-- Run the overlay process')
    recombine_keys = pylib.run_overlay(splits, cache)
    print(f'   {len(recombine_keys)} recombine keys')
    
    print('-- Simplify lines')
    simplified_keys = pylib.simplify_lines(cache, recombine_keys)
    print(f'   {len(simplified_keys)} simplified keys')
    
    print('-- Write the roads files')
    pylib.write_files(pkg, simplified_keys)
    
ns.add_task(create_roads_files)

@task
def create_points_files(c):
    """Build the geohash_tags.csv file"""
    cache_dir = str(Path(__file__).parent.resolve())
    print(f"Cache: {cache_dir}")
    
    pkg = mp.open_package(cache_dir)
    
    cache = pylib.open_cache(pkg)

    print('-- Convert PBF file')
    convert_pbf(c)
    
    print('-- Make tags dataframe')
    tags_df = pylib.make_tags_df(pkg)
    
    print('-- Extract class Columns')
    cls_df = pylib.extract_class_columns(tags_df)
    
    print('-- Make geotags dataframe')
    ght = pylib.make_geotags_df(pkg, tags_df, cls_df)
    
ns.add_task(create_points_files)

@task( optional=['force'])
def build(c, force=None):
    """Build a filesystem package."""

    create_roads_files(c)
    create_points_files(c)
    mp_build(c, force)

    
ns.add_task(build)
