import torch
import numpy as np
import random

from typing import Any, Sequence, Tuple, Union, Callable


class CopyPaste:

    def __init__(self, transforms: Union[None, Sequence[Callable]] = None, **kwargs) -> None:
        # set defaults
        self.area_ratio_range = (0.05, 0.15)  # ratio of patch-area:img-area
        self.aspect_ratio_range = (0.2, 5)  # aspect ratio of patch (w / h)
        self.transforms = transforms if transforms else None

        if 'area_ratio_range' in kwargs:
            area_ratio_range = kwargs['area_ratio_range']
            assert isinstance(area_ratio_range, tuple), f'{type(area_ratio_range)} is not accepted. Use a Tuple.'
            assert len(area_ratio_range) == 2, f'Tuple must be of len 2. Got {len(area_ratio_range)}'
            assert area_ratio_range[0] < area_ratio_range[1], f'Lower bound must come first.'

            self.area_ratio_range = kwargs['area_ratio_range']
        if 'aspect_ratio_range' in kwargs:
            aspect_ratio_range = kwargs['aspect_ratio_range']
            assert isinstance(aspect_ratio_range, tuple), f'{type(aspect_ratio_range)} is not accepted. Use a Tuple.'
            assert len(aspect_ratio_range) == 2, f'Tuple must be of len 2. Got {len(aspect_ratio_range)}'
            assert aspect_ratio_range[0] < aspect_ratio_range[1], f'Lower bound must come first.'

            self.area_ratio_range = aspect_ratio_range

    def _area_ratio(self) -> float:
        """Find a random area ratio within specified range"""
        return random.uniform(self.area_ratio_range[0], self.area_ratio_range[1])

    def _aspect_ratio(self) -> float:
        """Find a random aspect ratio within specified range"""
        return random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])

    def _apply_transforms(self, patch: torch.Tensor) -> torch.Tensor:
        """Applies transformations in the order given to the constructor

        Intented use is to apply transforms to a patch before pasting.

        Note
        ----
        These transforms should only include color shifts and flips.
        It is inadvisable to use rotations since this would create artefacts
        in the current implementation of 'paste'.
        """
        for t in self.transforms:
            patch = t(patch)

        return patch

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply a random copy and paste transform.

        A randomly sized patch is copied from a random location in img
        and pasted at another random location in new_img.

        Parameters
        ----------
        img: torch.Tensor of shape (3, H, W)
            original image

        Returns
        --------
        new_img: torch.Tensor of shape (3, H, W)
            tensor of the same dimensions as img
        """

        # find shape
        _, h, w = img.shape
        area_ratio = self._area_ratio()
        aspect_ratio = self._aspect_ratio()
        wp = int(np.sqrt(area_ratio * w * h * aspect_ratio))
        hp = int(wp / aspect_ratio)

        # find locations
        copy_loc = (
            random.randint(0, h - hp),
            random.randint(0, w - wp)
        )

        paste_loc = (
            random.randint(0, h - hp),
            random.randint(0, w - wp)
        )

        # cut
        patch = CopyPaste.copy(img, copy_loc, (hp, wp))

        if self.transforms:
            patch = self._apply_transforms(patch)

        # paste
        new_img = CopyPaste.paste(img, patch, paste_loc)

        return new_img

    @staticmethod
    def copy(img: torch.Tensor, loc: Tuple[int, int], shape: Tuple[int, int]) -> torch.Tensor:
        """Returns a cropped patch from an image.

        Parameters
        ----------
        img: torch.Tensor of shape (3, H, W)
            original image
        loc: (y, x)
            location of top left corner of crop box
        shape: (Hp, Wp)
            dimensions of the patch to copy

        Returns
        --------
        cropped view of img
        """
        return img[:, loc[0]:(loc[0] + shape[0]), loc[1]:(loc[1] + shape[1])]

    @staticmethod
    def paste(img: torch.Tensor, patch: torch.Tensor, loc: Tuple[int, int], inplace=False) -> torch.Tensor:
        """Paste a patch over an image

        Parameters
        ----------
        img:  torch.Tensor of shape (3, H, W)
            original image
        patch: torch.Tensor of shape (3, Hp, Wp)
            patch to paste of shape
        loc: (y, x)
            location of top left corner of the patch in img
        """
        _, h, w = img.shape
        _, hp, wp = patch.shape

        assert (loc[0] + hp) < h, "patch is too tall, or location is too close to the bottom border."
        assert (loc[1] + wp) < w, "patch is too wide, or location is too close to the right border."

        if inplace:
            img[:, loc[0]:(loc[0] + hp), loc[1]:(loc[1] + wp)] = patch
            return img

        new_img = img.clone()
        new_img[:, loc[0]:(loc[0] + hp), loc[1]:(loc[1] + wp)] = patch
        return new_img


class FancyPCA:
    """Performs the color augmentation from AlexNet [1]

    Colors are shifted along their principle component axes rather than being transformed
    with random noise.

    Parameters
    ----------
    img:  torch.Tensor of shape (3, H, W)
        original image

    Returns
    -------
    transformed: torch.Tensor of shape (3, H, W)
        transformed image

    [1] Krizhevsky et al. ImageNet Classification with Deep Convolutional Neural Networks
        https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """

    def __init__(self, alpha_dist: Tuple = None) -> None:
        # TODO: check shape of arguments
        self.alpha_dist = alpha_dist if alpha_dist else (0, 0.5)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape

        transformed = img.clone()

        # flatten into rgb vectors
        rgb = img.reshape(3, -1)

        # center
        rgb = rgb - rgb.mean(axis=1)[:, np.newaxis]

        # covariance between color channels
        rgb_cov = np.cov(rgb, rowvar=True)

        # eigen decomposition
        eig_vals, eig_vecs = np.linalg.eigh(rgb_cov)

        # scale eigenvalues
        alpha = np.random.normal(0, 0.5, size=[1, 3])
        scaled_eigval = alpha * eig_vals

        # creative shifting vector (weighted sum of eigenvectors)
        v = eig_vecs @ scaled_eigval.T

        # apply transformation
        transformed = (transformed.reshape(3, -1) + v).reshape(3, h, w)

        return transformed