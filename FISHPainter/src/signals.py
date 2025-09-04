import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, binary_erosion
from  .process_boxes import merge_boxes_by_labels
from numbers import Number

MAX_ITERS=100

# debug
import matplotlib.pyplot as plt
from cellplot.patches import draw_boxes_on_patch

from typing import Union

def create_signal(image_size, position, signal_size, signal_intensity=1):
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
    signal[signal < 0.2] = 0.
    signal *= signal_intensity

    return signal

def create_cluster(image_size, position, valid_positions, signal_size, cluster_size, signal_intensity):
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
    
    signal[position] = np.random.uniform(signal_intensity[0], signal_intensity[1], 1)[0]

    
    curr_position = position
    curr_signals = 0
    iters = 0
    while curr_signals < cluster_size:
        
        curr_signal_size = np.random.uniform(signal_size[0], signal_size[1], 1)[0]
        curr_signal_intensity = np.random.uniform(signal_intensity[0], signal_intensity[1], 1)[0]
                
        iters += 1
        if iters > MAX_ITERS:
            break
        
        next_position = np.array([np.round(x + np.random.choice([-2.5*curr_signal_size, +2.5*curr_signal_size], 1)[0], 0).astype(int) for x in curr_position])
        
        if not (np.any(np.all(valid_positions == next_position, axis=1))):
            continue
        
        signal[next_position[0], next_position[1]] = curr_signal_intensity
        
        curr_position = next_position
        curr_signals += 1
        
    max_intensity = np.max(signal)
    
    signal = gaussian_filter(signal, sigma=np.mean(signal_size))
    
    signal = signal / signal.max()
    signal[signal < 0.2] = 0.

    signal *= max_intensity

    return signal


def create_FISH(
    patch, 
    mask, 
    num_red=0, 
    num_green=0, 
    num_green_cluster=0, 
    num_red_cluster=0, 
    signal_size: Union[list[Number], int, float]=1, 
    signal_intensity: list[Number]=1,
    green_cluster_size=0, 
    red_cluster_size=0, 
    seed=None, 
    alpha=20, 
    sigma=2, 
    return_as_dict=False,
    merge_bboxes=False):
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
    
    
    if not isinstance(signal_size, Number):
        max_signal_size = max(signal_size)
    else:
        max_signal_size = signal_size
        signal_size = [signal_size, signal_size]
        
    if isinstance(signal_intensity, Number):
        signal_intensity = [signal_intensity, signal_intensity]
        
    # Erode mask
    structure = np.ones((int(6*max_signal_size), int(6*max_signal_size)))
    diluted_mask = binary_erosion(mask, structure=structure)
    
    valid_positions = np.argwhere(diluted_mask)

    bboxes, labels = [], []
        
    for _ in range(num_red):
        
        curr_size = np.random.uniform(signal_size[0], signal_size[1], 1)
        curr_intensity = np.random.uniform(signal_intensity[0], signal_intensity[1], 1)
        
        position = valid_positions[np.random.choice(len(valid_positions))]
        red_signal = create_signal((h, w), (position[0], position[1]), [curr_size, curr_size], curr_intensity)
        
        red_signal, bbox = elastic_transform(red_signal, alpha=alpha, sigma=sigma)
        patch[..., 0] += red_signal  # Add to red channel
        bboxes.append(bbox)
        labels.append(1)
        
        
    for _ in range(num_green):
        
        curr_size = np.random.uniform(signal_size[0], signal_size[1], 1)
        curr_intensity = np.random.uniform(signal_intensity[0], signal_intensity[1], 1)
        
        position = valid_positions[np.random.choice(len(valid_positions))]
        green_signal = create_signal((h, w), (position[0], position[1]), [curr_size, curr_size], curr_intensity)
        green_signal, bbox = elastic_transform(green_signal, alpha=alpha, sigma=sigma)
        patch[..., 1] += green_signal  # Add to green channel
        bboxes.append(bbox)
        labels.append(2)
        
        
    for _ in range(num_green_cluster):
        
        start_position = valid_positions[np.random.choice(len(valid_positions))]
        green_signal = create_cluster((h, w), (start_position[0], start_position[1]), valid_positions, signal_size, green_cluster_size, signal_intensity)
        green_signal, bbox = elastic_transform(green_signal, alpha=alpha, sigma=sigma)
        patch[..., 1] += green_signal  # Add to green channel
        bboxes.append(bbox)
        labels.append(3)
        
        
    for _ in range(num_red_cluster):
        
        start_position = valid_positions[np.random.choice(len(valid_positions))]
        red_signal = create_cluster((h, w), (start_position[0], start_position[1]), valid_positions, signal_size, red_cluster_size, signal_intensity)
        red_signal, bbox = elastic_transform(red_signal, alpha=alpha, sigma=sigma)
        patch[..., 0] += red_signal  # Add to red channel
        bboxes.append(bbox)
        labels.append(4)


    patch = np.clip(patch, 0, 1)  # Ensure values are in [0, 1] range
    
    #plt.imshow(draw_boxes_on_patch(patch.copy(), bboxes, labels)); plt.show()
    
    if merge_bboxes:
        bboxes, labels = merge_boxes_by_labels(bboxes, labels)
    
    #plt.imshow(draw_boxes_on_patch(patch.copy(), bboxes, labels)); plt.show()

    if return_as_dict:
        return dict(patch=patch, bboxes=bboxes, labels=labels)
    
    return patch, bboxes, labels


def elastic_transform(image, alpha, sigma):
    """
    Apply elastic transformation on an image.

    Parameters:
    - image: np.ndarray
        Numpy array containing the image.
    - alpha: float
        Scaling factor for the elastic transformation.
    - sigma: float
        Standard deviation for Gaussian smoothing of random displacement fields.

    Returns:
    - Transformed image as a numpy array.
    - Bounding box (min_x, min_y, max_x, max_y) as a tuple.
    """
    max_scale = image.max()
    random_state = np.random.RandomState(None)

    shape = image.shape
    # Generate displacement fields for spatial dimensions only
    dx = gaussian_filter((random_state.rand(*shape[-2:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[-2:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create a meshgrid for spatial dimensions
    x_grid, y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y_grid + dy, (-1, 1)), np.reshape(x_grid + dx, (-1, 1))

    # Apply transformation
    distorted_image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)

    # Optional Gaussian smoothing
    distorted_image = gaussian_filter(distorted_image, sigma=1)

    # Calculate bounding box of active region
    active_coords = np.argwhere(distorted_image > 0.1 * distorted_image.max())
    if active_coords.size == 0:
        return distorted_image, None  # Handle edge case with no active pixels
    min_y, min_x = active_coords.min(axis=0)
    max_y, max_x = active_coords.max(axis=0)

    # Normalize and scale
    normalized_image = (distorted_image / distorted_image.max() if distorted_image.max() > 0 else distorted_image) * max_scale

    return normalized_image, (min_x, min_y, max_x, max_y)