# cuML-Go

A production-oriented Go wrapper around the NVIDIA RAPIDS cuML C++ API using cgo.

This project provides native Go bindings for GPU-accelerated machine learning algorithms while maintaining an idiomatic Go API. The library supports both GPU-backed implementations through RAPIDS cuML and optimized native CPU implementations written in C++.

Currently implemented models:

- Random Forest (Classification)
- K-Nearest Neighbors (KNN)
- K-Means Clustering

---

# Architecture

The project follows a layered architecture in which each layer has a single responsibility.

```
                          USER APPLICATION
                                  │
                                  ▼
                 Go Public API (RF / KNN / KMeans)
                                  │
                                  ▼
                          Internal Go Packages
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
 internal/dataset           internal/matrix           internal/capi
        │                         │                         │
        └─────────────────────────┴─────────────────────────┘
                                  │
                             cgo Boundary
──────────────────────────────────────────────────────────────────────
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                                                   │
        ▼                                                   ▼
                    Native CPU Implementations      RAPIDS GPU Implementations
                               │                                 │
                               ▼                                 ▼
                        Custom C++ Code                  RAFT / cuML / cuBLAS
                               │                                 │
                               ▼                                 ▼
                             OpenMP                          CUDA Runtime
                               │                                 │
                               ▼                                 ▼
                              CPU                                GPU
```

The Go user never interacts directly with C, C++, CUDA, or RAPIDS.

---

# Project Structure

```
cuml-go/
│
├── benchmark/
│   ├── data/
│   └── results/
│
├── inference_compare/
│   ├── random_forest/
│   ├── knn/
│   └── kmeans/
│
├── cpp/
│   ├── random_forest/
│   │   ├── random_forest.cpp
│   │   ├── random_forest.h
│   │   ├── random_forest_cpu.cpp
│   │   └── random_forest_cpu.h
│   │
│   ├── knn/
│   │   ├── knn.cpp
│   │   ├── knn.h
│   │   ├── knn_cpu.cpp
│   │   └── knn_cpu.h
│   │
│   └── kmeans/
│       ├── kmeans.cpp
│       ├── kmeans.h
│       ├── kmeans_cpu.cpp
│       └── kmeans_cpu.h
│
├── go/
│   ├── random_forest/
│   ├── knn/
│   └── kmeans/
│
├── internal/
│   ├── benchmark/
│   ├── capi/
│   ├── csv/
│   ├── dataset/
│   ├── matrix/
│   ├── result/
│   └── timer/
│
├── scripts/
│   ├── build.sh
│   ├── configure.sh
│   └── env.sh
│
├── benchmark/
├── environment.yml
├── requirements.txt
├── conda_packages.txt
├── toolchain_versions.txt
├── CMakeLists.txt
└── go.mod
```

---

# Layer Responsibilities

## 1. Go Model Packages

```
go/random_forest/
go/knn/
go/kmeans/
```

This is the public API exposed to Go applications.

Example:

```go
rf, _ := randomforest.New()

rf.Fit(X, y)

predictions, _ := rf.Predict(X)

rf.Close()
```

Responsibilities:

- User-facing API
- Hyperparameter configuration
- Input validation
- Backend abstraction
- Resource management

No C or C++ code exists in this layer.

---

## 2. internal/dataset

```
internal/dataset/
```

Responsible for:

- CSV parsing
- Dataset loading
- Feature extraction
- Label extraction
- Type conversion

Output:

```
[][]float32

[]int
```

---

## 3. internal/matrix

```
internal/matrix/
```

Responsible for converting Go data structures into contiguous memory suitable for C and C++.

Example:

```
[][]float32

↓

[]float32
```

Utilities include:

- Matrix flattening
- Label conversion
- C integer conversion
- Memory layout conversion

This package performs memory layout conversion only.

---

## 4. internal/capi

```
internal/capi/
```

This package is the only Go package that directly uses cgo.

Responsibilities:

- Converting Go slices to C pointers
- Calling C functions
- Converting C output back to Go
- Error handling

Example:

```
Go slice

↓

*C.float

↓

kmeans_fit(...)
```

Everything below this point is no longer Go.

---

## 5. cgo Boundary

This is where execution leaves the Go runtime.

```
Go

↓

cgo

↓

C
```

Only the `internal/capi` package crosses this boundary.

---

## 6. C Wrapper

Located in:

```
cpp/<model>/
```

Example:

```cpp
extern "C" {

KMeansHandle* kmeans_create(...);

int kmeans_fit(...);

int kmeans_predict(...);

}
```

Responsibilities:

- Stable ABI
- C-compatible interface
- Callable from Go

This layer contains no machine learning logic.

---

## 7. CPU Implementations

Located in:

```
cpp/random_forest/random_forest_cpu.*
cpp/knn/knn_cpu.*
cpp/kmeans/kmeans_cpu.*
```

Responsibilities:

- Native C++ implementations
- OpenMP parallelization
- BLAS acceleration
- Distance-matrix optimization
- Heap optimization

These implementations do not depend on RAPIDS.

---

## 8. GPU Implementations

Located in:

```
cpp/random_forest/random_forest.*
cpp/knn/knn.*
cpp/kmeans/kmeans.*
```

Responsibilities:

- RAFT handle management
- GPU memory management
- RAPIDS integration
- CUDA execution

---

## 9. NVIDIA RAPIDS Stack

External dependencies:

```
libcuml
libraft
CUDA Runtime
CUDA Driver
cuBLAS
OpenBLAS
```

These libraries are external dependencies and are not modified.

---

# Ownership of Code

## Custom Code

Everything inside:

```
go/

internal/

cpp/

benchmark/

inference_compare/

scripts/

CMakeLists.txt
```

was written specifically for this project.

---

## External Code

Everything provided by:

```
libcuml

libraft

CUDA

OpenBLAS
```

belongs to the respective upstream projects.

---

# Data Flow

```
CSV Dataset

↓

LoadCSV()

↓

[][]float32
[]int

↓

RandomForest.Fit()

↓

matrix.Flatten()

↓

internal/capi

↓

C API

↓

C++ Wrapper

↓

RAPIDS or CPU Backend

↓

Prediction
```

Prediction follows the same pipeline in reverse.

---

# Installation

## 1. Install Miniforge

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

bash Miniforge3-Linux-x86_64.sh
```

Restart the terminal.

Verify:

```bash
conda --version
```

---

## 2. Clone the Repository

```bash
git clone <repository-url>

cd cuml-go
```

---

## 3. Create the Conda Environment

```bash
conda env create -f environment.yml
```

List environments:

```bash
conda env list
```

Activate the environment:

```bash
conda activate rapids-gcc14
```

If the environment name differs, inspect:

```bash
head environment.yml
```

---

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Install System Dependencies

```bash
sudo apt update

sudo apt install \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev
```

---

## 6. Verify CUDA

```bash
nvidia-smi

nvcc --version
```

---

## 7. Load the RAPIDS Environment

```bash
source scripts/env.sh
```

Current environment variables:

```bash
export OPENBLAS_NUM_THREADS=1

export LD_LIBRARY_PATH="$PWD/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

---

## 8. Configure

```bash
./scripts/configure.sh
```

---

## 9. Build

```bash
./scripts/build.sh
```

Generated artifacts are placed in:

```
build/
```

Including:

```
build/libcumlgo.so
```

---

# Running Inference Comparison Tests

Random Forest:

```bash
go test -v ./inference_compare/random_forest
```

KNN:

```bash
go test -v ./inference_compare/knn
```

KMeans:

```bash
go test -v ./inference_compare/kmeans
```

Limit dataset size:

```bash
go test -v ./inference_compare/knn -args --rows=100000
```

---

# Running Benchmarks

Execute:

```bash
go run ./go
```

Run on a subset:

```bash
go run ./go -rows=5000
```

Results are automatically written to:

```
benchmark/results/
```

Example output:

```
go_benchmark_210726155230.csv
```

---

# CSV Output

Generated CSV files contain:

```
Model
Backend
TrainRows
PredictionRows
Accuracy
PredictionTime(ms)
Throughput(samples/sec)
CPUUsage(%)
```

---

# Currently Implemented Models

## Random Forest

```go
rf, _ := randomforest.New()

rf.Fit(X, y)

predictions, _ := rf.Predict(X)

rf.Close()
```

---

## KNN

```go
knn, _ := knn.New()

knn.Fit(X, y)

predictions, _ := knn.Predict(X)

knn.Close()
```

---

## KMeans

```go
km, _ := kmeans.New()

km.Fit(X)

labels, _ := km.Predict(X)

km.Close()
```

---

# Extending the Library

To add a new model:

1. Add the C wrapper.

```
cpp/new_model/
```

2. Implement the CPU backend.

3. Implement the GPU backend.

4. Add the cgo wrapper.

```
internal/capi/
```

5. Add the public Go package.

```
go/new_model/
```

6. Add inference comparison tests.

```
inference_compare/new_model/
```

7. Add benchmark support.

No other parts of the project need to be modified.

---

# Design Goals

- Native Go API
- Zero Python dependency during execution
- Thin cgo layer
- Minimal wrapper overhead
- CPU and GPU backends
- Modular architecture
- Production-oriented resource management
- Easy extension for future models
- Direct access to RAPIDS C++ libraries

---

# Environment Reproducibility

The repository includes the following environment files:

| File | Purpose |
| --- | --- |
| `environment.yml` | Conda environment definition |
| `requirements.txt` | Python dependencies |
| `conda_packages.txt` | Complete Conda package list |
| `toolchain_versions.txt` | Compiler, CMake, and CUDA versions |

These files allow the development environment to be recreated on another machine.