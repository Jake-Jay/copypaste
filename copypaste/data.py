import torch
import numpy as np

from pathlib import Path
from skimage import io
from typing import Callable, Sequence

from copypaste.transforms import CopyPaste


class MVTecDataset:
    """MVTec [1] anomoly detection dataset with optional copy-paste.
        
    Parameters
    ----------
    path_to_dataset : str
        Path to root of test/train dataset.
    copypaste: bool, optional
        Whether to apply a copy paste transformation and return both the 
        original and transformed image.
    transforms: list[callable], optional
        Transforms that can be applied to patches which before pasting. 
        This will only be applied if `copypaste == True`.

    [1] https://www.mvtec.com/company/research/datasets
    """

    def __init__(
        self, 
        path_to_dataset: str, 
        copypaste: bool = False, 
        transforms: Sequence[Callable] = None,
        **kwargs
    ) -> None:
        

        # check if copypaste should be applied
        self.copypaste = copypaste
        if self.copypaste:
            # TODO include optional transforms here
            self.cp = CopyPaste(transforms=transforms, **kwargs)

        # get list of images
        root = Path(path_to_dataset)
        self.dataset_type = root.parts[-1]            
        
        if self.dataset_type == "train":
            self.dataset_dir = root / 'good'
        elif self.dataset_type == "test":
            # TODO handle this differently
            self.dataset_dir = root / 'good'
        else:
            raise ValueError(f'Provide a path to a test/train/val dataset.'
                            f'Provided: {path_to_dataset}')
        

        self.image_paths = list(self.dataset_dir.glob('*.png'))
    
    def __getitem__(self, idx):
        """get img at idx or if `copypaste==True` (img, CopyPaste(img))

        Parameters
        ----------
        idx : int
            index of image in dataset

        Returns
        -------
        img, Tuple(img, CopyPaste(img)
            img has shape (C, H, W) - torch images have their channel first
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = torch.from_numpy(io.imread(self.image_paths[idx]))
        img = img / img.max() # range[0,1]
        img = img.moveaxis(-1, 0) # shape (3, H, W)

        if self.copypaste:
            return img, self.cp(img)
        
        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def __repr__(self) -> str:
        return (f'MVTec Dataset ({len(self)} items): {self.dataset_dir}\n'
                f'mode: {self.dataset_type}\n'
                f'copypaste: {self.copypaste}')

if __name__ == "__main__":

    ds = MVTecDataset("/Users/jake/bayer/copypaste/data/toothbrush/train/")
    print(ds)
        