import os
import shutil
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple

import numpy as np

from PIL import Image


class MaSTr1325Dataset(VisionDataset):
    """
    """

    # url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"

    def __init__(
            self,
            root: str,
            image_set: str = "train",
            download: bool = False,
            transforms: Optional[Callable] = None,
    ) -> None:

        super(MaSTr1325Dataset, self).__init__(root, transforms)

        Top_Img_Path = '/content/MaSTr1325_images'
        Top_Mask_Path = '/content/MaSTr1325_masks'
        
        self.images = sorted(
            [
                os.path.join(Top_Img_Path, fname)
                for fname in os.listdir(Top_Img_Path)
                if fname.endswith(".jpg")
            ]
        )
        self.masks = sorted(
            [
                os.path.join(Top_Mask_Path , fname)
                for fname in os.listdir(Top_Mask_Path )
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )

        assert (len(self.images) == len(self.masks))

    # https://github.com/pytorch/vision/blob/11268ca79e9a54b12fdafdde558e809698f09b05/torchvision/datasets/voc.py#L153-L167
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        
        
#         Obstacles and environment = 0 (value zero)
#         Water = 1 (value one)
#         Sky = 2 (value two)
#         Ignore region / unknown category = 4 (value four)


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Image set: {image_set}", "Mode: {mode}"]
        return '\n'.join(lines).format(**self.__dict__)
