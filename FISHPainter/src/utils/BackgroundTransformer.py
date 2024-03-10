
import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


class BackgroundTransformer(Dataset):
    def __init__(self, dataset_path):

        self.dataset = h5py.File(dataset_path, "r")

        self.transforms = T.Compose([
            T.ToTensor(),  # Converts numpy arrays to PyTorch tensors
            T.RandomAffine(degrees=(1, 360), scale=(0.75, 1.5))  # Applies random rotations and scaling
        ])

    def __getitem__(self, idx):

        patch = np.atleast_3d(self.dataset["rgb_background"][idx]).astype(np.float32)
        mask = np.atleast_3d(self.dataset["cell_masks"][idx]).astype(np.float32)
        
        combined = np.concatenate([patch, mask], axis=2)
        transformed = self.transforms(combined)

        patch_transformed = transformed[:3, :, :].numpy().transpose(1, 2, 0)  # First three channels for the background
        mask_transformed = transformed[-1, :, :].numpy()  # Last channel for the mask

        return patch_transformed, mask_transformed

    def __len__(self):

        return self.dataset["cell_masks"].shape[0]