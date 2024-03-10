import h5py
import numpy as np
from pathlib import Path

def print_structure(nested_dict, level=0):
    for key, value in nested_dict.items():
        # Construct the current path
        print("     "*level + key)
        # If the value is another dictionary, recurse with the updated path
        if isinstance(value, dict):
            print_structure(value, level=level+1)

def save2h5(outdict, h5_filepath):
    
    images, masks, parameters, classes = [], [], [], []
    for k, v in outdict.items():
        
        images.extend(v["rgb_patches"])
        masks.extend(v["masks"])
        parameters.extend(v["parameters"])
        classes.extend(v["target_classes"])
        
    images = np.array(images)
    masks = np.array(masks)
    parameters = np.array(parameters)
    classes = np.array(classes)
    
    if Path(h5_filepath).exists():
        raise Exception(f"{h5_filepath} already exists!")
    
    with h5py.File(h5_filepath, 'w') as f:
        
        f.create_dataset("image_patches", data=images)
        f.create_dataset("mask_patches", data=masks)
        f.create_dataset("parameters", data=parameters)
        f.create_dataset("classes", data=classes)
