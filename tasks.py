# Task definitions for invoke
# You must first install invoke, https://www.pyinvoke.org/

import fiona # Avoids a bizzare error: AttributeError: module 'fiona' has no attribute '_loading'
import sys
from pathlib import Path
import metapack as mp
from invoke import task
from metapack_build.tasks.package import ns, build as mp_build
from metapack.appurl import SearchUrl

SearchUrl.initialize()  # This makes the 'index:" urls work
sys.path.append(str(Path(__file__).parent.resolve()))

import pylib
import importlib
importlib.reload(pylib) # Because when using a collection with invoke, may already have been loaded

import logging
from pylib import lines_logger, points_logger, cluster_logger

sys.path.pop() # Other other datasets wont get this pylib

logging.basicConfig()
lines_logger.setLevel(logging.INFO)
points_logger.setLevel(logging.INFO)
cluster_logger.setLevel(logging.INFO)

pbf_file = 'north-america-latest.osm.pbf'
pbf_url = f'https://download.geofabrik.de/{pbf_file}'

# You can also create you own tasks


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

    dd = Path.cwd().joinpath('data')

    if not dd.exists():
        dd.mkdir(parents=True )

    if not dd.joinpath(pbf_file).exists():
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
        return
    
    if not p.exists():
        points_logger.error("ERROR: Download https://download.geofabrik.de/north-america-latest.osm.pbf and put it in the data directory"
              " or run `invoke get_pbf`")
        
        sys.exit(1)
    
    if out_dir.exists():
        print(f"ERROR: output dir {out_dir} should not exist")
        sys.exit(1)

    points_logger.info('Convert PBF file')
    op.convert_pbf(p, out_dir)

ns.add_task(convert_pbf)

@task
def create_roads_files(c):
    """Build the residential_roads.csv and nonres_roads.csv files"""
    cache_dir = str(Path(__file__).parent.resolve())
    lines_logger.info(f"Cache: {cache_dir}")
    
    pkg = mp.open_package(cache_dir)

    convert_pbf(c)

    pylib.build_lines(pkg)

    
ns.add_task(create_roads_files)

@task
def create_points_files(c):
    """Build the geohash_tags.csv file"""
    
    pkg_dir = str(Path(__file__).parent.resolve())
    pkg = mp.open_package(pkg_dir)
    points_logger.info(f"Pkg dir: {pkg_dir}")

    convert_pbf(c)
    
    pylib.build_points(pkg)
    
ns.add_task(create_points_files)

@task
def build_osm_blocks(c):
    """Build blocks geo file and assign OSM points to blocks"""
    pkg_dir = str(Path(__file__).parent.resolve())
    pkg = mp.open_package(pkg_dir)
    points_logger.info(f"Pkg dir: {pkg_dir}")

    pylib.build_osm_points(pkg)
ns.add_task(build_osm_blocks)




@task
def build_block_maps(c):

    pkg_dir = str(Path(__file__).parent.resolve())
    pkg = mp.open_package(pkg_dir)
    points_logger.info(f"Pkg dir: {pkg_dir}")

    pylib.build_block_maps(pkg)
ns.add_task(build_block_maps)

@task
def build_clusters(c):

    pkg_dir = str(Path(__file__).parent.resolve())
    pkg = mp.open_package(pkg_dir)
    points_logger.info(f"Pkg dir: {pkg_dir}")

    pylib.build_clusters(pkg)

ns.add_task(build_clusters)


@task( optional=['force'])
def build(c, force=None):
    """Build a filesystem package."""

    create_roads_files(c)
    create_points_files(c)
    build_osm_blocks(c)
    build_block_maps(c)

    mp_build(c, force)

    
ns.add_task(build)
