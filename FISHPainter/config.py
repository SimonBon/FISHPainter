from pathlib import Path
import os
from .src.utils.zenodo_download import download_from_zenodo

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