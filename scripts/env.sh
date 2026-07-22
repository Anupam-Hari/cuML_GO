#!/usr/bin/env bash

conda activate rapids-cpp

export LD_LIBRARY_PATH="$PWD/build:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"