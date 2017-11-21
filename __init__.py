import os

_ROOT = neuralbinder.__path__[0]
def get_datasets(path):
    return os.path.join(_ROOT, 'datasets', path)
