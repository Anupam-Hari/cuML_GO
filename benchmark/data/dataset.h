#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct Dataset
{
    std::vector<float> X;
    std::vector<int> y;

    int rows = 0;
    int cols = 0;

    std::unordered_map<int, std::string> class_names;
};