import h5py

def dataset_from_bboxes(bbox_dataset, filepath, train_split=0.9, val_split=0.1):
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
    assert train_split + val_split == 1.0, "Train and validation split should sum to 1.0"

    num_samples = len(bbox_dataset)
    num_train = int(train_split * num_samples)
    num_val = num_samples - num_train

    with h5py.File(filepath, 'w') as f:
        train_group = f.create_group("training")
        val_group = f.create_group("validation")

        for n, data in enumerate(bbox_dataset):
            if n < num_train:
                g = train_group.create_group(str(n))
            else:
                g = val_group.create_group(str(n - num_train))

            g.create_dataset("patch", data=data["patch"])
            g.create_dataset("labels", data=data["labels"])
            g.create_dataset("bboxes", data=data["bboxes"])
