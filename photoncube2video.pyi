from typing import List, Optional
from typing_extensions import Self

import numpy as np

class PhotonCube:
    path: str
    cfa_mask: Optional[np.ndarray]
    inpaint_mask: Optional[np.ndarray]
    start: int
    end: Optional[int]
    step: Optional[int]

    @classmethod
    def open(path: str) -> Self: ...
    @staticmethod
    def convert_to_npy(
        src: str,
        dst: str,
        is_full_array: bool,
        message: Optional[str],
    ) -> None: ...
    def load_cfa(self: Self, path: str) -> None: ...
    def load_mask(self: Self, path: str) -> None: ...
    def set_range(
        self: Self, start: int, end: Optional[int] = None, step: Optional[int] = None
    ) -> None: ...
    def set_transforms(self: Self, transforms: List[str]) -> None: ...
    def save_images(
        img_dir,
        invert_response=False,
        tonemap2srgb=False,
        colorspad_fix=False,
        annotate_frames=False,
        message=None,
    ) -> int: ...
    def save_video(
        output,
        fps=24,
        img_dir=None,
        invert_response=False,
        tonemap2srgb=False,
        colorspad_fix=False,
        annotate_frames=False,
        message=None,
    ) -> int: ...
    def __len__(self: Self) -> int: ...
