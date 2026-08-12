#!/usr/bin/env bash

conda activate rapids-gcc14

export LD_LIBRARY_PATH="$PWD/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OPENBLAS_NUM_THREADS=1