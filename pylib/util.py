from pathlib import Path

from demosearch import FileCache


def get_cache(pkg):
    return FileCache(Path(pkg.path).parent.joinpath('data', 'cache'))


