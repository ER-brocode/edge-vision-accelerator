import pytest
import numpy as np
from edge_vision_accelerator.inference.detector import Detection, QuantizedYOLODetector, MobileNetDetector
from edge_vision_accelerator.inference.engine import InferenceEngine, EngineConfig, ExecutionBackend


def test_detection_iou_calculation():
    d1 = Detection(bbox=[0, 0, 10, 10], class_id=0, confidence=0.9, label="person")
    d2 = Detection(bbox=[0, 0, 10, 10], class_id=0, confidence=0.8, label="person")
    d3 = Detection(bbox=[20, 20, 30, 30], class_id=0, confidence=0.7, label="person")

    assert d1.iou(d2) == pytest.approx(1.0)
    assert d1.iou(d3) == pytest.approx(0.0)

    # Partial overlap
    d4 = Detection(bbox=[5, 0, 15, 10], class_id=0, confidence=0.85, label="person")
    # intersection: 5 * 10 = 50. union = 100 + 100 - 50 = 150. iou = 50/150 = 1/3
    assert d1.iou(d4) == pytest.approx(1.0 / 3.0)


def test_mock_inference_engine_execution():
    config = EngineConfig(backend=ExecutionBackend.MOCK)
    engine = InferenceEngine(model_path=None, config=config)

    fake_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    output = engine.run({"images": fake_input})

    assert len(output) == 1
    assert output[0].shape == (1, 84, 8400)


def test_quantized_yolo_detector_mock_tensor_pipeline():
    detector = QuantizedYOLODetector(model_path=None, conf_threshold=0.3, nms_threshold=0.5)

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob, ratio, pad = detector.preprocess(img)

    assert blob.shape == (1, 3, 640, 640)
    assert 0.0 <= blob.max() <= 1.0

    # Inject mock tensor with two high-confidence bounding boxes
    mock_raw = np.zeros((1, 84, 8400), dtype=np.float32)
    # Box 1: cx=320, cy=240, w=100, h=100, class_0 conf=0.95
    mock_raw[0, 0, 10] = 320.0
    mock_raw[0, 1, 10] = 240.0
    mock_raw[0, 2, 10] = 100.0
    mock_raw[0, 3, 10] = 100.0
    mock_raw[0, 4, 10] = 0.95  # Person

    # Box 2 (duplicate box to test NMS suppression):
    mock_raw[0, 0, 11] = 322.0
    mock_raw[0, 1, 11] = 241.0
    mock_raw[0, 2, 11] = 98.0
    mock_raw[0, 3, 11] = 99.0
    mock_raw[0, 4, 11] = 0.85  # Lower conf person

    # Box 3: different location
    mock_raw[0, 0, 20] = 100.0
    mock_raw[0, 1, 20] = 100.0
    mock_raw[0, 2, 20] = 50.0
    mock_raw[0, 3, 20] = 50.0
    mock_raw[0, 6, 20] = 0.90  # Car (class 2)

    detections = detector.postprocess([mock_raw], ratio, pad, (480, 640))
    # Duplicate box 2 should be suppressed by Box 1
    assert len(detections) == 2
    assert detections[0].confidence == pytest.approx(0.95)
    assert detections[0].label == "person"
    assert detections[1].label == "car"


def test_mobilenet_detector_postprocessing():
    detector = MobileNetDetector(model_path=None, conf_threshold=0.4)

    # Format [1, N, 7]: [batch, class_id, score, x1, y1, x2, y2]
    mock_raw = np.zeros((1, 1, 2, 7), dtype=np.float32)
    mock_raw[0, 0, 0] = [0, 0, 0.88, 0.1, 0.1, 0.3, 0.4]  # class 0: person
    mock_raw[0, 0, 1] = [0, 2, 0.20, 0.5, 0.5, 0.8, 0.8]  # low score

    dets = detector.postprocess([mock_raw[0]], 1.0, (0, 0), (480, 640))
    assert len(dets) == 1
    assert dets[0].label == "person"
    assert dets[0].bbox[0] == pytest.approx(64.0)
    assert dets[0].bbox[1] == pytest.approx(48.0)
