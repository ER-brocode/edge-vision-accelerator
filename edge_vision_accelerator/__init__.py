"""Edge Vision Accelerator (EVA) - High Performance Edge AI Stream Processor."""

__version__ = "1.0.0"

from edge_vision_accelerator.pipeline.frame import Frame, FrameMeta, PixelFormat
from edge_vision_accelerator.pipeline.ring_buffer import ZeroCopyFrameRingBuffer
from edge_vision_accelerator.pipeline.ingestion import VideoStreamIngestor, RTSPStreamIngestor, V4L2CameraIngestor
from edge_vision_accelerator.inference.detector import BaseDetector, QuantizedYOLODetector, MobileNetDetector
from edge_vision_accelerator.inference.engine import ExecutionBackend, InferenceEngine
from edge_vision_accelerator.tracker.byte_track import ByteTracker
from edge_vision_accelerator.tracker.deep_sort import DeepSORTTracker
from edge_vision_accelerator.tracker.kalman import KalmanBoxTracker

__all__ = [
    "Frame",
    "FrameMeta",
    "PixelFormat",
    "ZeroCopyFrameRingBuffer",
    "VideoStreamIngestor",
    "RTSPStreamIngestor",
    "V4L2CameraIngestor",
    "BaseDetector",
    "QuantizedYOLODetector",
    "MobileNetDetector",
    "ExecutionBackend",
    "InferenceEngine",
    "ByteTracker",
    "DeepSORTTracker",
    "KalmanBoxTracker",
]
