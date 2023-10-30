import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, binary_erosion
from  .process_boxes import merge_boxes_by_labels

def create_signal(image_size, position, signal_size, min_brightness=0.2, max_brightness=1.0):
    """
    Creates a Gaussian signal of specified size at a specified position.

    Parameters:
    - image_size: Tuple indicating the size of the output image (height, width).
    - position: Tuple indicating the position (y, x) to place the Gaussian center.
    - signal_size: Size (standard deviation) of the Gaussian.

    Returns:
    - A 2D numpy array representing the image with the Gaussian signal.
    - Bounding box coordinates as a tuple (min_y, min_x, max_y, max_x).
    """
    signal = np.zeros(image_size)
    signal[position] = 1
    signal = gaussian_filter(signal, sigma=signal_size)
    signal = signal / signal.max()
    signal[signal > 0.8] = 1.
    signal[signal < 0.8] = 0.
    
    scaling_factor = np.random.uniform(min_brightness, max_brightness)
    signal *= scaling_factor

    # Obtain bounding box coordinates

    return signal


def create_FISH(patch, mask, num_red=2, num_green=2, signal_size=3, seed=None, alpha=20, sigma=2, return_as_dict=False):
    """
    Modifies the input image patch by placing Gaussian dots based on the mask and desired number of red and green dots.

    Parameters:
    - patch: Input image patch of shape (height, width, 3).
    - mask: Binary mask indicating valid positions for placing dots.
    - num_red: Number of red Gaussian dots to place.
    - num_green: Number of green Gaussian dots to place.
    - signal_size: Size (standard deviation) of the Gaussian dots.
    - seed: Optional random seed for reproducibility.

    Returns:
    - A numpy array representing the modified image patch.
    """
    
    # Validation for spatial dimensions of mask and patch
    if patch.shape[:2] != mask.shape:
        raise ValueError("Spatial dimensions of mask and patch do not match.")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    h, w, _ = patch.shape
    
    # Erode mask
    structure = np.ones((int(signal_size), int(signal_size)))
    diluted_mask = binary_erosion(mask, structure=structure)
    
    valid_positions = np.argwhere(diluted_mask)

    bboxes, labels = [], []
    

    for _ in range(num_red):
        position = valid_positions[np.random.choice(len(valid_positions))]
        red_signal = create_signal((h, w), (position[0], position[1]), signal_size)
        red_signal, bbox = elastic_transform(red_signal, alpha=alpha, sigma=sigma)
        patch[..., 0] += red_signal  # Add to red channel
        bboxes.append(bbox)
        labels.append(1)
        
    for _ in range(num_green):
        position = valid_positions[np.random.choice(len(valid_positions))]
        green_signal = create_signal((h, w), (position[0], position[1]), signal_size)
        green_signal, bbox = elastic_transform(green_signal, alpha=alpha, sigma=sigma)
        patch[..., 1] += green_signal  # Add to green channel
        bboxes.append(bbox)
        labels.append(2)

    patch = np.clip(patch, 0, 1)  # Ensure values are in [0, 1] range
    bboxes, labels = merge_boxes_by_labels(bboxes, labels)
    
    if return_as_dict:
        return dict(patch=patch, bboxes=bboxes, labels=labels)
    
    return patch, bboxes, labels



def elastic_transform(image, alpha, sigma):
    """
    Apply elastic transformation on an image.
    
    Parameters:
    - image: Numpy array containing the image.
    - alpha: Scaling factor.
    - sigma: Elasticity coefficient.

    Returns:
    - Transformed image as a numpy array.
    """
    
    max_scale = image.max()
    
    random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)
    distored_image = gaussian_filter(distored_image, sigma=1)
    
    active_coords = np.argwhere(distored_image > 0.1*distored_image.max())
    min_y, min_x = active_coords.min(axis=0)
    max_y, max_x = active_coords.max(axis=0)
    
    return (distored_image/distored_image.max())*max_scale, (min_y, min_x, max_y, max_x)