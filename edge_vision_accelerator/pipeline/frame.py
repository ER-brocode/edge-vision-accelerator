"""Frame and metadata representation with zero-copy array views."""

from dataclasses import dataclass
from enum import Enum
import time
import numpy as np


class PixelFormat(str, Enum):
    BGR = "bgr"
    RGB = "rgb"
    RGBA = "rgba"
    GRAY = "gray"
    NV12 = "nv12"


@dataclass
class FrameMeta:
    frame_id: int
    timestamp_ns: int
    width: int
    height: int
    channels: int
    format: PixelFormat
    stride: int
    camera_id: str = "default"


class Frame:
    """Zero-copy friendly Frame wrapper pointing to underlying memory buffers."""

    def __init__(self, data: np.ndarray, meta: FrameMeta):
        self._data = data
        self.meta = meta

    @property
    def data(self) -> np.ndarray:
        """Returns the numpy array without making a deep copy."""
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def width(self) -> int:
        return self.meta.width

    @property
    def height(self) -> int:
        return self.meta.height

    @property
    def frame_id(self) -> int:
        return self.meta.frame_id

    @property
    def timestamp_ns(self) -> int:
        return self.meta.timestamp_ns

    def to_rgb(self) -> np.ndarray:
        if self.meta.format == PixelFormat.RGB:
            return self._data
        elif self.meta.format == PixelFormat.BGR:
            return self._data[:, :, ::-1]
        elif self.meta.format == PixelFormat.GRAY:
            return np.repeat(self._data[:, :, np.newaxis], 3, axis=2)
        return self._data

    def to_bgr(self) -> np.ndarray:
        if self.meta.format == PixelFormat.BGR:
            return self._data
        elif self.meta.format == PixelFormat.RGB:
            return self._data[:, :, ::-1]
        return self._data

    @classmethod
    def from_numpy(cls, arr: np.ndarray, frame_id: int = 0, format: PixelFormat = PixelFormat.BGR, camera_id: str = "default") -> "Frame":
        h, w = arr.shape[:2]
        c = arr.shape[2] if arr.ndim == 3 else 1
        stride = arr.strides[0]
        meta = FrameMeta(
            frame_id=frame_id,
            timestamp_ns=time.time_ns(),
            width=w,
            height=h,
            channels=c,
            format=format,
            stride=stride,
            camera_id=camera_id
        )
        return cls(arr, meta)
