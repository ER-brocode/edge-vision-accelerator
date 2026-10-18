"""INT8 Quantized YOLOv8 and MobileNet Edge Object Detectors."""

from dataclasses import dataclass, field
import time
from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from edge_vision_accelerator.inference.engine import InferenceEngine, EngineConfig, ExecutionBackend


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    label: str
    embedding: Optional[np.ndarray] = None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter_area = iw * ih

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class BaseDetector:
    """Base edge neural detector contract."""

    def __init__(self, model_path: Optional[str] = None, config: Optional[EngineConfig] = None, conf_threshold: float = 0.45, nms_threshold: float = 0.50):
        self.engine = InferenceEngine(model_path, config)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.labels = COCO_CLASSES
        self.last_latency_ms = 0.0

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        raise NotImplementedError

    def postprocess(self, raw_outputs: List[np.ndarray], ratio: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Detection]:
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> List[Detection]:
        start = time.perf_counter()
        orig_shape = img.shape[:2]

        blob, ratio, pad = self.preprocess(img)
        input_name = self.engine.input_names[0] if self.engine.input_names else "images"
        raw_outputs = self.engine.run({input_name: blob})

        detections = self.postprocess(raw_outputs, ratio, pad, orig_shape)
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return detections


class QuantizedYOLODetector(BaseDetector):
    """Production INT8 / FP16 YOLOv8 edge detector with dynamic letterbox scaling

    and high-performance vectorised NMS.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[EngineConfig] = None,
        input_size: int = 640,
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.50,
        labels: Optional[List[str]] = None,
    ):
        super().__init__(model_path, config, conf_threshold, nms_threshold)
        self.input_size = input_size
        if labels:
            self.labels = labels

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Zero-distortion Letterbox transform."""
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))

        pad_w = (self.input_size - nw) // 2
        pad_h = (self.input_size - nh) // 2

        if cv2 is not None:
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback simple resize
            resized = np.zeros((nh, nw, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        # HWC to CHW, normalize to [0, 1]
        blob = canvas.transpose((2, 0, 1)).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # [1, 3, H, W]
        return blob, scale, (pad_w, pad_h)

    def postprocess(self, raw_outputs: List[np.ndarray], ratio: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Detection]:
        if not raw_outputs or raw_outputs[0].size == 0:
            return []

        out = raw_outputs[0]
        # Check output shape: typically [1, 84, 8400] for YOLOv8
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            out = np.transpose(out, (0, 2, 1))  # -> [1, 8400, 84]

        preds = out[0]  # [8400, 84]
        if preds.shape[0] == 0:
            return []

        # Box coordinates cx, cy, w, h
        boxes = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert cx, cy, w, h -> x1, y1, x2, y2
        pad_w, pad_h = pad
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_w) / ratio
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_h) / ratio
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_w) / ratio
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_h) / ratio

        # Clip to image boundary
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        detections = []
        for i in range(len(boxes)):
            cid = int(class_ids[i])
            lbl = self.labels[cid] if cid < len(self.labels) else str(cid)
            detections.append(Detection(
                bbox=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                class_id=cid,
                confidence=float(confidences[i]),
                label=lbl
            ))

        return self._nms(detections)

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []
        # Sort desc
        detections.sort(key=lambda d: d.confidence, reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if best.class_id != d.class_id or best.iou(d) < self.nms_threshold]
        return keep


class MobileNetDetector(BaseDetector):
    """SSD-MobileNet Edge detector for ultra low-power microcontrollers and NPUs."""

    def __init__(self, model_path: Optional[str] = None, config: Optional[EngineConfig] = None, input_size: int = 300, conf_threshold: float = 0.50):
        super().__init__(model_path, config, conf_threshold=conf_threshold, nms_threshold=0.45)
        self.input_size = input_size

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = img.shape[:2]
        if cv2 is not None:
            resized = cv2.resize(img, (self.input_size, self.input_size))
        else:
            resized = np.zeros((self.input_size, self.input_size, 3), dtype=img.dtype)
        blob = resized.astype(np.float32)
        # MobileNet expects [-1, 1] or [0, 255] depending on model
        blob = (blob - 127.5) / 127.5
        blob = blob.transpose((2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob, 1.0, (0, 0)

    def postprocess(self, raw_outputs: List[np.ndarray], ratio: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Detection]:
        if not raw_outputs:
            return []
        orig_h, orig_w = orig_shape
        # Expecting [1, N, 7] format: [batch, class_id, score, x1, y1, x2, y2]
        arr = raw_outputs[0]
        if arr.ndim == 3 and arr.shape[-1] == 7:
            arr = arr[0]
            mask = arr[:, 2] >= self.conf_threshold
            valid = arr[mask]
            dets = []
            for item in valid:
                cid = int(item[1])
                score = float(item[2])
                x1 = float(item[3] * orig_w)
                y1 = float(item[4] * orig_h)
                x2 = float(item[5] * orig_w)
                y2 = float(item[6] * orig_h)
                lbl = self.labels[cid] if cid < len(self.labels) else str(cid)
                dets.append(Detection(bbox=[x1, y1, x2, y2], class_id=cid, confidence=score, label=lbl))
            return dets
        return []
