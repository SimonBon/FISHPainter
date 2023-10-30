import h5py

def dataset_from_bboxes(bbox_dataset, filepath):

    with h5py.File(filepath, 'w') as f:
        for n, data in enumerate(bbox_dataset):
            g = f.create_group(str(n))
            g.create_dataset("patches", data=data["patch"])
            g.create_dataset("labels", data=data["labels"])
            g.create_dataset("bboxes", data=data["bboxes"])