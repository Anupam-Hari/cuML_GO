#include "dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

Dataset load_csv(const std::string& filename, bool has_header)
{
    std::ifstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Unable to open file: " + filename);

    Dataset dataset;

    std::string line;

    // Skip header if present
    if (has_header)
        std::getline(file, line);

    std::unordered_map<std::string, int> label_map;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);

        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ','))
            tokens.push_back(token);

        if (tokens.size() < 2)
            continue;

        // Number of features
        if (dataset.cols == 0)
            dataset.cols = static_cast<int>(tokens.size()) - 1;

        // Features
        for (int i = 0; i < dataset.cols; ++i)
            dataset.X.push_back(std::stof(tokens[i]));

        // Label (last column)
        const std::string& label = tokens.back();

        auto it = label_map.find(label);

        if (it == label_map.end())
        {
            int id = static_cast<int>(label_map.size());
            label_map[label] = id;
            dataset.y.push_back(id);
        }
        else
        {
            dataset.y.push_back(it->second);
        }

        dataset.rows++;
    }

    dataset.classes = static_cast<int>(label_map.size());

    return dataset;
}

std::vector<float> transpose(
        const std::vector<float>& X,
        int rows,
        int cols)
    {
        std::vector<float> out(rows * cols);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                out[c * rows + r] = X[r * cols + c];
            }
        }

        return out;
    }
