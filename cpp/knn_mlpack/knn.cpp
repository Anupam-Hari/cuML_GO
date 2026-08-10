#include "knn.h"

#include <mlpack/methods/neighbor_search/knn.hpp>

#include <armadillo>

#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>

struct KNNHandle
{
    std::unique_ptr<mlpack::KNN> model;

    arma::fmat train;

    arma::Row<size_t> labels;

    int k;

    int rows;
    int cols;

    KNNHandle()
        : k(0),
          rows(0),
          cols(0)
    {
    }
};

KNNHandle* knn_create(int k)
{
    auto* knn = new KNNHandle;

    knn->k = k;

    return knn;
}

int knn_fit(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    const int* y)
{
    try
    {
        handle->rows = rows;
        handle->cols = cols;

        //-------------------------------------------------
        // Copy training data into Armadillo
        //-------------------------------------------------

        handle->train.set_size(cols, rows);

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                handle->train(c, r) = X[r * cols + c];
            }
        }

        //-------------------------------------------------
        // Copy labels
        //-------------------------------------------------

        handle->labels.set_size(rows);

        for (int i = 0; i < rows; ++i)
        {
            handle->labels[i] = static_cast<size_t>(y[i]);
        }

        //-------------------------------------------------
        // Build KNN model
        //-------------------------------------------------

        handle->model =
            std::make_unique<mlpack::KNN>(handle->train);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "knn_fit exception: "
                  << e.what() << std::endl;

        return -1;
    }
    catch (...)
    {
        std::cerr << "knn_fit unknown exception"
                  << std::endl;

        return -1;
    }
}

int knn_predict(
    KNNHandle* handle,
    const float* X,
    int rows,
    int cols,
    int* predictions)
{
    try
    {
        if (cols != handle->cols)
            return -1;

        //-------------------------------------------------
        // Convert query matrix
        //-------------------------------------------------

        arma::fmat query(cols, rows);

        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                query(c, r) = X[r * cols + c];
            }
        }

        //-------------------------------------------------
        // Search
        //-------------------------------------------------

        arma::Mat<size_t> neighbors;
        arma::fmat distances;

        handle->model->Search(
            query,
            handle->k,
            neighbors,
            distances
        );

        //-------------------------------------------------
        // Majority vote
        //-------------------------------------------------

        for (int q = 0; q < rows; ++q)
        {
            std::unordered_map<int, int> votes;

            for (int i = 0; i < handle->k; ++i)
            {
                size_t trainIndex = neighbors(i, q);

                int label =
                    static_cast<int>(handle->labels[trainIndex]);

                votes[label]++;
            }

            int bestLabel = -1;
            int bestVotes = -1;

            for (const auto& kv : votes)
            {
                if (kv.second > bestVotes)
                {
                    bestVotes = kv.second;
                    bestLabel = kv.first;
                }
            }

            predictions[q] = bestLabel;
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "knn_predict exception: "
                  << e.what() << std::endl;

        return -1;
    }
    catch (...)
    {
        std::cerr << "knn_predict unknown exception"
                  << std::endl;

        return -1;
    }
}

void knn_destroy(KNNHandle* handle)
{
    if (handle == nullptr)
        return;

    handle->model.reset();

    delete handle;
}