#include "dataset_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace
{
std::vector<std::string> split(const std::string& line)
{
    std::stringstream ss(line);

    std::vector<std::string> tokens;

    std::string token;

    while (std::getline(ss, token, ','))
        tokens.push_back(token);

    return tokens;
}
}

float parse_feature(const std::string& value)
{
    if (value == "True")
        return 1.0f;

    if (value == "False")
        return 0.0f;

    return std::stof(value);
}

Dataset load_dataset(
    const std::string& filename,
    const std::string& label_column,
    int max_rows)
{
    std::ifstream file(filename);

    if (!file.is_open())
        throw std::runtime_error("Unable to open dataset: " + filename);

    Dataset dataset;

    std::string line;

    if (!std::getline(file, line))
        throw std::runtime_error("Dataset is empty.");

    std::vector<std::string> header = split(line);

    int label_index = -1;

    int other_label_index = -1;

    for (int i = 0; i < static_cast<int>(header.size()); ++i)
    {
        if (header[i] == label_column)
            label_index = i;

        if (label_column == "is_malicious" &&
            header[i] == "attack_type")
            other_label_index = i;

        if (label_column == "attack_type" &&
            header[i] == "is_malicious")
            other_label_index = i;
    }

    if (label_index == -1)
        throw std::runtime_error("Label column not found.");

    std::unordered_map<std::string, int> label_encoder;

    while (std::getline(file, line))
    {
        if (max_rows > 0 && dataset.rows >= max_rows)
            break;
        
        if (line.empty())
            continue;

        auto values = split(line);

        for (int i = 0; i < static_cast<int>(values.size()); ++i)
        {
            if (i == label_index || i == other_label_index)
                continue;

            try
            {
                dataset.X.push_back(parse_feature(values[i]));
            }
            catch (const std::exception&)
            {
                std::cerr << "Invalid float at row "
                        << dataset.rows
                        << ", column "
                        << i
                        << " ("
                        << header[i]
                        << ") = '"
                        << values[i]
                        << "'"
                        << std::endl;

                throw;
            }
        }

        if (label_column == "is_malicious")
        {
            dataset.y.push_back(std::stoi(values[label_index]));
        }
        else
        {
            auto it = label_encoder.find(values[label_index]);

            if (it == label_encoder.end())
            {
                int id = static_cast<int>(label_encoder.size());

                label_encoder[values[label_index]] = id;

                dataset.class_names[id] = values[label_index];

                dataset.y.push_back(id);
            }
            else
            {
                dataset.y.push_back(it->second);
            }
        }

        dataset.rows++;
    }

    dataset.cols =
        static_cast<int>(dataset.X.size()) / dataset.rows;

    std::cout << "Loaded dataset\n";
    std::cout << "Rows     : " << dataset.rows << '\n';
    std::cout << "Features : " << dataset.cols << '\n';

    if (label_column == "attack_type")
    {
        std::cout << "Classes  : "
                  << dataset.class_names.size()
                  << '\n';
    }

    return dataset;
}