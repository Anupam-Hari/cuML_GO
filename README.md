# cuML-Go

A production-oriented Go wrapper around the NVIDIA RAPIDS cuML C++ API using cgo. This project provides native Go bindings for GPU-accelerated machine learning algorithms while keeping the Go API simple and idiomatic.

Currently implemented models:

- Random Forest (Classification)
- K-Nearest Neighbors (KNN)
- K-Means Clustering

---
## Build Requirements

* Ubuntu 22.04+
* NVIDIA GPU with compatible NVIDIA driver
* Go **1.26.6**
* CMake **4.4.0**
* GCC/G++ **14.3.0**
* CUDA **12.9**
* NVIDIA RAPIDS/cuML **26.06.00**
* mlpack **4.8.1**, source commit: c5ebb4655737a1fc17a5071e539d4e1d17f5b740
* Treelite **4.7.0**
* RAFT **26.06.00**
* RMM **26.06.00**
* Armadillo/OpenBLAS/OpenMP as native dependencies

### Setup

```bash
# Create the RAPIDS environment
conda env create -f environment.yml
conda activate rapids-gcc14

# Verify the toolchain
go version
cmake --version
gcc --version
g++ --version
nvcc --version
pkg-config --modversion mlpack

# Configure
./configure.sh

# Build
./build.sh

# Run tests
go test ./...
```

### Runtime

A production deployment requires:

* Prebuilt Go application
* `libcumlgo.so`
* Saved model files
* NVIDIA driver
* CUDA/RAPIDS runtime libraries
* mlpack runtime libraries for CPU inference

Python is **not required at runtime**.

# Architecture

The project follows a layered architecture where each layer has a single responsibility.

```
                USER APPLICATION
                      │
                      ▼
          Go Public API (random_forest, knn, kmeans)
                      │
                      ▼
            Go Internal C API Wrapper (capi)
                      │
                 cgo Boundary
──────────────────────────────────────────────────────────
                      │
                      ▼
               C Wrapper (extern "C")
                      │
                      ▼
             C++ Wrapper (Custom Code)
                      │
                      ▼
          NVIDIA RAPIDS cuML C++ Library
                      │
                      ▼
                CUDA Runtime / GPU
```

The Go user never interacts directly with C, C++, CUDA, or cuML.

---

# Project Structure

```
cuml-go/
│
├── benchmark/
│   ├── benchmark.cpp
│   ├── benchmark.h
│   ├── main.cpp
│   ├── timer.h
│   ├── result.h
│   ├── dataset_loader.*
│   ├── data/
│   └── results/
│
├── cpp/
│   ├── random_forest/
│   ├── knn/
│   └── kmeans/
|
├── scripts/
│   ├── build.sh
│   ├── configure.sh
│
├── go/
│   ├── main.go
│   ├── benchmark.go
│   ├── dataset.go
│   ├── timer.go
│   ├── result.go
│   ├── csv.go
│   │
│   ├── internal/
│   │   ├── capi/
│   │   ├── matrix/
│   │   └── dataset/
│   │
│   ├── random_forest/
│   ├── knn/
│   └── kmeans/
│
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

This is the public API exposed to Go users.

Example:

```go
rf, _ := randomforest.New()

rf.Fit(X, y)

predictions, _ := rf.Predict(X)
```

Responsibilities:

- User-friendly API
- Hyperparameter configuration
- Input validation
- Data conversion
- Resource management

No C or C++ code exists here.

---

## 2. internal/capi

```
go/internal/capi/
```

This package is the only Go package that directly uses cgo.

Responsibilities:

- Calls C functions
- Converts Go slices to C pointers
- Converts C output back to Go
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

## 3. cgo Boundary

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

## 4. C Wrapper

Located in

```
cpp/<model>/*.h
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

- C-compatible interface
- Stable ABI
- Callable from Go

This layer contains **no machine learning logic**.

---

## 5. C++ Wrapper

Located in

```
cpp/random_forest/
cpp/knn/
cpp/kmeans/
```

Responsibilities:

- Convert C API into C++ classes
- Manage RAFT handles
- Allocate GPU resources
- Call cuML

This layer contains the custom C++ implementation.

---

## 6. NVIDIA RAPIDS cuML

Everything below this layer belongs to NVIDIA.

Examples:

- Random Forest
- KNN
- KMeans
- RAFT
- CUDA kernels
- GPU memory management

This project simply calls these APIs.

No modifications are made to NVIDIA code.

---

# Ownership of Code

## Custom Code

Everything in:

```
go/

cpp/

benchmark/

CMakeLists.txt
```

is written specifically for this project.

---

## NVIDIA Code

Everything inside installed RAPIDS libraries:

```
libcuml

libraft

CUDA Runtime

CUDA Driver

cuBLAS

etc.
```

These are external dependencies.

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

matrix.From2D()

↓

internal/capi

↓

C API

↓

C++ Wrapper

↓

cuML

↓

GPU
```

Prediction follows the same pipeline in reverse.

---

# internal/matrix

Purpose:

Convert Go data structures into contiguous memory suitable for C.

Examples:

```
[][]float32

↓

[]float32
```

Also contains utilities such as

- NumClasses()
- ToCInt()
- FromCInt32()

This package performs **memory layout conversion only**.

---

# internal/dataset

Responsible for

- CSV parsing
- Boolean conversion
- Label extraction
- Feature loading

Output:

```
[][]float32

[]int
```

---

# internal/capi

Contains wrappers like

```
RandomForestCreate()

RandomForestFit()

RandomForestPredict()

KNNCreate()

KNNFit()

KNNPredict()

KMeansCreate()

KMeansFit()

KMeansPredict()
```

These functions directly invoke C.

---

# Build Instructions

## 1. Activate the RAPIDS Environment

Before building or running the project, activate and source the RAPIDS C++ conda environment.

```bash
source scripts/env.sh
```

---

## 2. Configure the Project

Run the provided configuration script from the project root.

```bash
./scripts/configure.sh
```

This configures the CMake build directory and locates the required RAPIDS, CUDA, and C++ dependencies.

---

## 3. Build

Compile the project using the build script.

```bash
./scripts/build.sh
```

This builds the shared library and all C++ executables.

The generated artifacts are placed in the `build/` directory, including:

```
build/libcumlgo.so
```

which is used by the Go cgo wrappers.

---


# Running Go Tests

Random Forest

```bash
go test -v ./go/random_forest
```

KNN

```bash
go test -v ./go/knn
```

KMeans

```bash
go test -v ./go/kmeans
```

Limit dataset size

```bash
go test -v ./go/random_forest -args -rows=5000
```

---

# Currently Implemented Models

## Random Forest

Public API

```go
rf, _ := randomforest.New()

rf.Fit(X, y)

pred, _ := rf.Predict(X)

rf.Close()
```

---

## KNN

```go
knn, _ := knn.New()

knn.Fit(X, y)

pred, _ := knn.Predict(X)

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

To add a new cuML model:

1. Create a C wrapper

```
cpp/new_model/
```

2. Implement the C++ wrapper calling cuML.

3. Add a cgo wrapper

```
go/internal/capi/
```

4. Add a public Go package

```
go/new_model/
```

5. Write Go integration tests.

6. Add benchmark implementation.

No other parts of the project need modification.

---

## Porting to a New Machine

`cuml-go` consists of a Go layer, a native C++/CUDA shared library, and model files.

### Build Machine

A new build machine must provide the same native toolchain and library versions:

* Ubuntu 22.04+
* Go **1.26.6**
* CMake **4.4.0**
* GCC/G++ **14.3.0**
* CUDA **12.9**
* RAPIDS/cuML **26.06.00**
* Treelite **4.7.0**
* RAFT **26.06.00**
* RMM **26.06.00**
* mlpack **4.8.1**
* Required native dependencies such as Armadillo, OpenBLAS, and OpenMP

The RAPIDS dependencies can be recreated using `environment.yml`.

**mlpack note:** mlpack 4.8.1 is currently installed from source into the Conda environment rather than tracked by Conda. It is a **build-time dependency** for the CPU backend. The exact source commit used is:

```text
c5ebb4655737a1fc17a5071e539d4e1d17f5b740
```

For reproducible builds on a new machine, mlpack should be installed from this pinned source version (or vendored as a project dependency).

### Production Machine

A production machine running a **prebuilt** application does not need the Go compiler, CMake, GCC/G++, or mlpack source/headers.

Deploy:

```text
Go application
libcumlgo.so
model files
model metadata (.meta)
```

The target machine must provide the required native runtime libraries and, for GPU inference, a compatible NVIDIA driver/GPU and CUDA/RAPIDS runtime.

**Python is not required. ONNX Runtime is not required.**

### Key Principle

```text
Build machine
    ↓
Go + C++/CUDA + RAPIDS + mlpack
    ↓
libcumlgo.so + Go executable
    ↓
Production machine
    ↓
Go executable + libcumlgo.so + model files + runtime libraries
```

mlpack is compiled into the native library through its C++ headers; it is not a separate runtime library that needs to be installed on the production machine.


# Design Goals

- Native Go API
- Zero Python dependency
- Thin cgo layer
- Minimal wrapper overhead
- Modular architecture
- Production-quality resource management
- Easy addition of future cuML algorithms
- Direct access to NVIDIA RAPIDS C++ libraries