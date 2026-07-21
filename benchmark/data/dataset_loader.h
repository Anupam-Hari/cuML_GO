#pragma once

#include <string>

#include "dataset.h"

Dataset load_dataset(
    const std::string& filename,
    const std::string& label_column,
    int max_rows = -1);