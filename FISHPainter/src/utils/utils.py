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

def save2h5(outdict, h5_filepath, manual_classes=None, overwrite=False):
    
    h5_filepath = Path(h5_filepath)
    
    images, masks, parameters, classes = [], [], [], []
    for k, v in outdict.items():
        
        images.extend(v["rgb_patches"])
        masks.extend(v["masks"])
        parameters.extend(v["parameters"])
        classes.extend(v["target_classes"])
        
    class_names = list(outdict.keys())
    class_numbers = [v['target_classes'][0] for k, v in outdict.items()]
    
    class_mapping_np = np.stack([class_names, class_numbers], axis=0).astype(h5py.string_dtype(encoding="utf-8"))
    
    images = np.array(images)
    masks = np.array(masks)
    parameters = np.array(parameters)
    classes = np.array(classes)

    if manual_classes is not None:
        classes = np.array(manual_classes)
    
    if h5_filepath.exists():
        if overwrite:
            h5_filepath.unlink()
        else:
            raise Exception(f"{h5_filepath} already exists! If you want to overwrite set 'overwrite=True'")
    

    with h5py.File(h5_filepath, 'w') as f:
        
        f.create_dataset("image_patches", data=images)
        f.create_dataset("mask_patches", data=masks)
        f.create_dataset("parameters", data=parameters)
        if not np.all(classes == None):
            f.create_dataset("class", data=classes)
        f.create_dataset('class_number_mapping', data=class_mapping_np)
        
        
class FISHDataset():
    
    def __init__(self, h5_path):
        
        
        self.h5_path = h5_path
        
        self.name2class_mapping, self.class2name_mapping = self.get_mapping()
        
    def get_mapping(self):
        
        with h5py.File(self.h5_path, 'r') as f:
            mapping = f['class_number_mapping'][:].astype(str).T
            
        name2class_mapping = {class_name: int(class_value) for (class_name, class_value) in mapping}
        class2name_mapping = {int(class_value): class_name for (class_name, class_value) in mapping}

        
        return name2class_mapping, class2name_mapping            
    
        
