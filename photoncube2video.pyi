from os import PathLike
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
    def open(path: PathLike) -> Self: ...
    @staticmethod
    def convert_to_npy(
        src: PathLike,
        dst: PathLike,
        is_full_array: bool = False,
        message: Optional[str] = None,
    ) -> None: ...
    def process_cube(
        self: Self,
        dst: PathLike,
        colorspad_fix: Optional[bool] = False,
        grayspad_fix: Optional[bool] = False,
        message: Optional[str] = None,
    ) -> Tuple[int]: ...
    def load_cfa(self: Self, path: PathLike) -> None: ...
    def load_mask(self: Self, path: PathLike) -> None: ...
    def set_range(
        self: Self,
        start: int = 0,
        end: Optional[int] = None,
        burst_size: Optional[int] = None,
    ) -> None: ...
    def set_transforms(self: Self, transforms: List[Transform]) -> None: ...
    def set_quantile(self: Self, quantile: Optional[float]) -> None: ...
    def save_images(
        img_dir: PathLike = None,
        invert_response: bool = False,
        tonemap2srgb: bool = False,
        colorspad_fix: bool = False,
        grayspad_fix: bool = False,
        annotate_frames: bool = False,
        message: Optional[str] = None,
        step: int = 1,
    ) -> int: ...
    def save_video(
        output: PathLike,
        fps: int = 24,
        img_dir: Optional[PathLike] = None,
        invert_response: bool = False,
        tonemap2srgb: bool = False,
        colorspad_fix: bool = False,
        grayspad_fix: bool = False,
        annotate_frames: bool = False,
        crf: int = 28,
        preset: str = "ultrafast",
        message: Optional[str] = None,
        step: int = 1,
    ) -> int: ...
    @property
    def shape(self: Self) -> Tuple[int]: ...
    def __len__(self: Self) -> int: ...
