#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "eva/types.hpp"
#include "eva/ring_buffer.hpp"
#include "eva/accelerator.hpp"
#include "eva/byte_track.hpp"

namespace py = pybind11;
using namespace eva;

PYBIND11_MODULE(_eva_cpp, m) {
    m.doc() = "Edge Vision AI Accelerator C++ native core module";

    py::enum_<PixelFormat>(m, "PixelFormat")
        .value("BGR24", PixelFormat::BGR24)
        .value("RGB24", PixelFormat::RGB24)
        .value("RGBA32", PixelFormat::RGBA32)
        .value("NV12", PixelFormat::NV12)
        .value("YUV420P", PixelFormat::YUV420P)
        .export_values();

    py::enum_<BackendDevice>(m, "BackendDevice")
        .value("CPU", BackendDevice::CPU)
        .value("CUDA", BackendDevice::CUDA)
        .value("TensorRT", BackendDevice::TensorRT)
        .value("OpenVINO", BackendDevice::OpenVINO)
        .value("NPU_DirectML", BackendDevice::NPU_DirectML)
        .export_values();

    py::class_<Rect>(m, "Rect")
        .def(py::init<float, float, float, float>(), py::arg("x")=0.0f, py::arg("y")=0.0f, py::arg("width")=0.0f, py::arg("height")=0.0f)
        .def_readwrite("x", &Rect::x)
        .def_readwrite("y", &Rect::y)
        .def_readwrite("width", &Rect::width)
        .def_readwrite("height", &Rect::height)
        .def("iou", &Rect::iou)
        .def("__repr__", [](const Rect& r) {
            return "<Rect x=" + std::to_string(r.x) + " y=" + std::to_string(r.y) +
                   " w=" + std::to_string(r.width) + " h=" + std::to_string(r.height) + ">";
        });

    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("bbox", &Detection::bbox)
        .def_readwrite("class_id", &Detection::class_id)
        .def_readwrite("confidence", &Detection::confidence)
        .def_readwrite("label", &Detection::label)
        .def_readwrite("embedding", &Detection::embedding);

    py::class_<TrackedObject>(m, "TrackedObject")
        .def(py::init<>())
        .def_readwrite("track_id", &TrackedObject::track_id)
        .def_readwrite("bbox", &TrackedObject::bbox)
        .def_readwrite("class_id", &TrackedObject::class_id)
        .def_readwrite("confidence", &TrackedObject::confidence)
        .def_readwrite("label", &TrackedObject::label)
        .def_readwrite("age_frames", &TrackedObject::age_frames)
        .def_readwrite("hits", &TrackedObject::hits)
        .def_readwrite("velocity", &TrackedObject::velocity);

    py::class_<ZeroCopyFrameRingBuffer>(m, "ZeroCopyFrameRingBuffer")
        .def(py::init<size_t, size_t>(), py::arg("capacity"), py::arg("max_frame_bytes"))
        .def("size", &ZeroCopyFrameRingBuffer::size)
        .def("capacity", &ZeroCopyFrameRingBuffer::capacity)
        .def("dropped_frames", &ZeroCopyFrameRingBuffer::dropped_frames)
        .def("reset", &ZeroCopyFrameRingBuffer::reset)
        .def("shutdown", &ZeroCopyFrameRingBuffer::shutdown);

    py::class_<ByteTrackerCpp>(m, "ByteTrackerCpp")
        .def(py::init<int, float, float, float>(),
             py::arg("max_lost_frames") = 30,
             py::arg("high_thresh") = 0.5f,
             py::arg("low_thresh") = 0.1f,
             py::arg("match_thresh") = 0.8f)
        .def("update", &ByteTrackerCpp::update)
        .def("reset", &ByteTrackerCpp::reset);
}
