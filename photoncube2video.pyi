from __future__ import annotations

from os import PathLike
from enum import Enum, auto
from typing import List, Tuple
from typing_extensions import Self

import numpy.typing as npt

class Transform(Enum):
    Identity = auto()
    Rot90 = auto()
    Rot180 = auto()
    Rot270 = auto()
    FlipUD = auto()
    FlipLR = auto()

    @classmethod
    def from_str(cls, transform_name: str) -> Self: ...

class PhotonCube:
    path: str
    cfa_mask: npt.NDArray | None
    inpaint_mask: npt.NDArray | None
    start: int
    end: int | None
    burst_size: int | None

    @classmethod
    def open(cls, path: PathLike) -> Self: ...
    @staticmethod
    def convert_to_npy(
        src: str | PathLike,
        dst: str | PathLike,
        is_full_array: bool = False,
        message: str | None = None,
    ) -> None: ...
    def process_cube(
        self: Self,
        dst: PathLike,
        colorspad_fix: bool = False,
        grayspad_fix: bool = False,
        message: str | None = None,
    ) -> Tuple[int]: ...
    def load_cfa(self: Self, path: PathLike) -> None: ...
    def load_mask(self: Self, path: PathLike) -> None: ...
    def set_range(
        self: Self,
        start: int = 0,
        end: int | None = None,
        burst_size: int | None = None,
    ) -> None: ...
    def set_transforms(self: Self, transforms: List[Transform]) -> None: ...
    def set_quantile(self: Self, quantile: float | None) -> None: ...
    def save_images(
        self: Self,
        img_dir: str | PathLike,
        invert_response: bool = False,
        tonemap2srgb: bool = False,
        colorspad_fix: bool = False,
        grayspad_fix: bool = False,
        annotate_frames: bool = False,
        message: str | None = None,
        step: int = 1,
    ) -> int: ...
    def save_video(
        self: Self,
        output: str | PathLike,
        fps: int = 24,
        img_dir: str | PathLike | None = None,
        invert_response: bool = False,
        tonemap2srgb: bool = False,
        colorspad_fix: bool = False,
        grayspad_fix: bool = False,
        annotate_frames: bool = False,
        crf: int = 28,
        preset: str = "ultrafast",
        message: str | None = None,
        step: int = 1,
    ) -> int: ...
    @property
    def shape(self: Self) -> Tuple[int]: ...
    def is_empty(self: Self) -> bool: ...
    def __len__(self: Self) -> int: ...
    def __getitem__(self: Self, idx: int) -> npt.NDArray: ...
