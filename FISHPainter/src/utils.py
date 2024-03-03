import h5py
import numpy as np

def create_dataset(dataset, filepath):
    """
    Create an h5py file and populate it with data from a bounding box dataset.
    The data is split into training and validation sets.

    Parameters:
        bbox_dataset: The bounding box dataset.
        filepath: The path to save the h5py file.
        train_split: The fraction of data to use for training (default is 0.9).
        val_split: The fraction of data to use for validation (default is 0.1).

    Returns:
        None
    """
    
    print(dataset.keys())

    with h5py.File(filepath, 'w') as f:
        
        for i in range(len(dataset["patches"])):
            
            g = f.create_group(str(i))

            g.create_dataset("patches", data=dataset["patches"][i])
            g.create_dataset("labels", data=dataset["labels"][i])
            g.create_dataset("bboxes", data=dataset["bboxes"][i])
