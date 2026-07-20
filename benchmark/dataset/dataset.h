#pragma once

#include <string>
#include <vector>

struct Dataset {
    std::vector<float> X;
    std::vector<int> y;

    int rows = 0;
    int cols = 0;
    int classes = 0;
};

Dataset load_csv(
    const std::string& filename,
    bool has_header = true
);
