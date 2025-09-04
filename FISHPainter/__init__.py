import os
from pathlib import Path
from .src.utils.zenodo_download import download_from_zenodo
from .src.datasets.create import create_dataset
from . import config


# Define the version of the script/package
__version__ = '0.5'