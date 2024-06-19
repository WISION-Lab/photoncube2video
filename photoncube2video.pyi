from enum import Enum, auto
from typing import List, Tuple, Optional
from typing_extensions import Self

import numpy as np

class Transform(Enum):
    Identity = auto()
    Rot90 = auto()
    Rot180 = auto()
    Rot270 = auto()
    FlipUD = auto()
    FlipLR = auto()

    def from_str(transform_name: str) -> Self: ...
    

class PhotonCube:
    path: str
    cfa_mask: Optional[np.ndarray]
    inpaint_mask: Optional[np.ndarray]
    start: int
    end: Optional[int]
    burst_size: Optional[int]

    @classmethod
    def open(path: str) -> Self: ...
    @staticmethod
    def convert_to_npy(
        src: str,
        dst: str,
        is_full_array: bool,
        message: Optional[str],
    ) -> None: ...
    def process_cube(
        self: Self,
        dst: str,
        colorspad_fix: Optional[bool] = False,
        grayspad_fix: Optional[bool] = False,
        message=None,
    ) -> Tuple[int]: ...
    def load_cfa(self: Self, path: str) -> None: ...
    def load_mask(self: Self, path: str) -> None: ...
    def set_range(
        self: Self, start: int, end: Optional[int] = None, burst_size: Optional[int] = None
    ) -> None: ...
    def set_transforms(self: Self, transforms: List[Transform]) -> None: ...
    def set_quantile(self: Self, quantile: Optional[float]) -> None: ...
    def save_images(
        img_dir,
        invert_response=False,
        tonemap2srgb=False,
        colorspad_fix=False,
        grayspad_fix=False,
        annotate_frames=False,
        message=None,
        step=1
    ) -> int: ...
    def save_video(
        output,
        fps=24,
        img_dir=None,
        invert_response=False,
        tonemap2srgb=False,
        colorspad_fix=False,
        grayspad_fix=False,
        annotate_frames=False,
        message=None,
        step=1
    ) -> int: ...
    def __len__(self: Self) -> int: ...
