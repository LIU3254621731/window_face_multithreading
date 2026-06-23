<h1 align="center">🫀 rPPG on RK3588</h1>

<p align="center">
  <strong>High-Performance Multi-Threaded Heart Rate Monitoring on ARM Edge Device</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B" alt="C++17">
  <img src="https://img.shields.io/badge/RK3588-ARM-FF6F00" alt="RK3588">
  <img src="https://img.shields.io/badge/RKNN-NPU-00BFFF" alt="RKNN">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/CMake-3.20-064F8C?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 📖 Overview

Production-grade C++ implementation of remote photoplethysmography (rPPG) heart rate monitoring optimized for the **RK3588 ARM platform** with NPU acceleration. Designed as a high-throughput embedded pipeline capable of real-time multi-face processing.

**Pipeline**: GStreamer → YuNet NPU Face Detection → 6-Region Parallel POS Algorithm → FFT Fusion → Heart Rate Output

## ✨ Features

### 🚀 High Performance
- **Multi-threaded** architecture with condition variables
- **RKNN NPU** acceleration for face detection inference
- **Eigen** linear algebra for optimized matrix operations
- **Parallel POS** algorithm across 6 facial sub-regions

### 🎯 Face Detection
- YuNet face detector with RKNN NPU backend
- RetinaFace alternative support
- Multi-face concurrent tracking

### 📊 Signal Processing
- POS (Plane-Orthogonal-to-Skin) rPPG algorithm
- IIR bandpass filtering (Butterworth, Chebyshev)
- FFT-based heart rate estimation
- Signal quality assessment

### 🔧 Cross-Platform Build
- CMake build system
- Windows (x86) development support
- RK3588 (ARM) deployment target
- WSL/Ubuntu verification

## 🏛️ Architecture

```
┌──────────────────────────────────────────────┐
│                  Main Thread                  │
│         Orchestrator + FFT Fusion            │
└───┬────────┬────────┬──────────┬────────────┘
    │        │        │          │
┌───▼──┐ ┌──▼──┐ ┌──▼───┐ ┌───▼──────┐
│Thread│ │Thread│ │Thread│ │  Thread   │
│ Cam  │ │ Face │ │ POS  │ │  Output   │
│ (GS) │ │(NPU) │ │ (x6) │ │           │
└──────┘ └─────┘ └──────┘ └──────────┘
```

## 🚀 Quick Start

### Prerequisites
- CMake 3.20+
- C++17 compiler
- OpenCV 4.x
- RKNN SDK (for RK3588 deployment)

### Build (x86/Windows)

```bash
cd cpp
cmake -B build -S .
cmake --build build --config Release
```

### Build (RK3588/ARM)

```bash
cd cpp
cmake -B build -S . \
  -DCMAKE_TOOLCHAIN_FILE=../scripts/rk3588_toolchain.cmake \
  -DRKNN_SDK=/path/to/rknn
cmake --build build
```

## 📁 Structure

```
window_face_multithreading/
├── cpp/                  # Core C++ implementation
│   ├── camera.cc/h       # Camera capture (GStreamer)
│   ├── yunet.cc/h        # YuNet face detector (NPU)
│   ├── pos_algorithm.*   # POS rPPG algorithm
│   ├── filter.*          # IIR signal filtering
│   ├── SignalProcessor.* # FFT + heart rate
│   ├── image_processor.* # Image preprocessing
│   └── main.cc           # Entry point
├── 3rdparty/             # Vendored dependencies
│   ├── Eigen/            # Linear algebra
│   ├── fftw/             # FFT library
│   ├── opencv/           # Computer vision
│   └── rknpu2/           # RKNN NPU SDK
├── model/                # Face detection models
├── python/               # Python utilities
└── docs/                 # Documentation
```

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | C++17 |
| Build | CMake 3.20+ |
| Vision | OpenCV 4.x |
| Face Detection | YuNet + RetinaFace |
| NPU | RKNN (RK3588) |
| Math | Eigen 3.4, FFTW |
| Signal | IIR (Butterworth/Chebyshev), POS, FFT |
| Camera | GStreamer |

## 📝 License

MIT
