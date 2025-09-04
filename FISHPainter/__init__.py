import os
from pathlib import Path
from .src.utils.zenodo_download import download_from_zenodo

# Define the DOI for the Zenodo download
DOI = "10798938"

# Safely access the HOME environment variable and construct the FISHPainter home directory path
home_path = Path(os.getenv("HOME", default=""))
if not home_path:
    raise EnvironmentError("HOME environment variable is not set.")

FISHPainter_home = home_path / ".FISHPainter"

# Check if the FISHPainter_home directory exists, create it if it does not, and then proceed to download
if not FISHPainter_home.exists():
    FISHPainter_home.mkdir()
    download_from_zenodo(DOI, str(FISHPainter_home))

# Define the version of the script/package
__version__ = '0.5'