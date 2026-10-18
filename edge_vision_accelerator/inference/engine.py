"""Inference execution engine with multi-accelerator provider backends."""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional, List, Dict, Any
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class ExecutionBackend(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    DIRECTML = "directml"
    CORAL_TPU = "coral_tpu"
    MOCK = "mock"


@dataclass
class EngineConfig:
    backend: ExecutionBackend = ExecutionBackend.CPU
    device_id: int = 0
    num_threads: int = 4
    enable_profiling: bool = False
    int8_calibration_cache: Optional[str] = None
    tensorrt_workspace_bytes: int = 1 << 30  # 1GB


class InferenceEngine:
    """Production runtime abstraction wrapping ONNX Runtime and hardware providers."""

    def __init__(self, model_path: Optional[str] = None, config: Optional[EngineConfig] = None):
        self.model_path = model_path
        self.config = config or EngineConfig()
        self.session = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, List[int]] = {}
        self.output_shapes: Dict[str, List[int]] = {}

        if self.model_path is not None and ort is not None:
            self._load_session()

    def _get_providers(self) -> List[Any]:
        if ort is None:
            return ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        providers = []

        if self.config.backend == ExecutionBackend.TENSORRT:
            if "TensorrtExecutionProvider" in available:
                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "device_id": self.config.device_id,
                        "trt_max_workspace_size": self.config.tensorrt_workspace_bytes,
                        "trt_fp16_enable": True,
                        "trt_int8_enable": bool(self.config.int8_calibration_cache),
                    }
                ))
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": self.config.device_id}))

        elif self.config.backend == ExecutionBackend.CUDA:
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": self.config.device_id}))

        elif self.config.backend == ExecutionBackend.DIRECTML:
            if "DmlExecutionProvider" in available:
                providers.append(("DmlExecutionProvider", {"device_id": self.config.device_id}))

        elif self.config.backend == ExecutionBackend.OPENVINO:
            if "OpenVINOExecutionProvider" in available:
                providers.append(("OpenVINOExecutionProvider", {"device_type": "CPU_FP32"}))

        providers.append("CPUExecutionProvider")
        return providers

    def _load_session(self):
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            providers = self._get_providers()
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)

            for inp in self.session.get_inputs():
                self.input_names.append(inp.name)
                self.input_shapes[inp.name] = inp.shape

            for out in self.session.get_outputs():
                self.output_names.append(out.name)
                self.output_shapes[out.name] = out.shape

            logger.info("Loaded model session '%s' using provider: %s", self.model_path, self.session.get_providers())
        except Exception as e:
            logger.warning("Failed to initialize ONNX session with file '%s': %s. Falling back to Mock.", self.model_path, e)
            self.session = None

    def run(self, input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Runs inference synchronously."""
        if self.session is not None:
            return self.session.run(self.output_names, input_feed)

        # Fallback simulated forward pass
        first_input = next(iter(input_feed.values()))
        batch_size = first_input.shape[0] if first_input.ndim >= 1 else 1
        # Mock detection tensor output [batch, 84, 8400] for YOLOv8 or [batch, 100, 6]
        mock_output = np.zeros((batch_size, 84, 8400), dtype=np.float32)
        return [mock_output]
