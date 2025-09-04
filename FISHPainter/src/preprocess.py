import tifffile
import numpy as np
from numba import jit

def get_cell_background(filepath: str, normalize: bool = False) -> np.ndarray:
    """
    Reads a TIFF image, sets its red and green channels to zero, and optionally normalizes the image.
    
    Parameters:
    - filepath: Path to the TIFF image.
    - normalize: Whether to scale the pixel values between 0 and 1.

    Returns:
    - A numpy array representing the image with red and green channels set to zero (and optionally normalized).
    """
    
    # Load image
    image = tifffile.imread(filepath)
    
    # Set the red and green channels to zero
    image[..., 0] = 0
    image[..., 1] = 0

    if normalize:
        normalize_image_inplace(image)

    return image 

@jit(nopython=True)
def normalize_image_inplace(image: np.ndarray) -> None:
    """
    Normalize the pixel values of an image between 0 and 1 in-place.

    Parameters:
    - image: Input image as a numpy array.
    """
    min_val = image.min()
    max_val = image.max()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                image[i, j, k] = (image[i, j, k] - min_val) / (max_val - min_val)
