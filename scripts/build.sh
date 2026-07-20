#!/usr/bin/env bash
set -e

cd build
cmake --build . -j$(nproc)
