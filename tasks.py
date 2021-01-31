# Task definitions for invoke
# You must first install invoke, https://www.pyinvoke.org/

from pathlib import Path

pbf_file = 'north-america-latest.osm.pbf'
pbf_url = f'https://download.geofabrik.de/{pbf_file}'

# You can also create you own tasks
from invoke import task

from metapack_build.tasks.package import ns

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
    
    if not p.exists():
        print("ERROR: Download https://download.geofabrik.de/north-america-latest.osm.pbf and put it in the data directory"
              " or run `invoke get_pbf`")
        
        sys.exit(1)
    
    if out_dir.exists():
        print(f"ERROR: output dir {out_dir} should not exist")
        sys.exit(1)
        
   
    op.convert_pbf(p, out_dir)


ns.add_task(convert_pbf)
