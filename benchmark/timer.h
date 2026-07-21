#pragma once

#include <chrono>

class Timer
{
public:
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop()
    {
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double, std::milli>(
                   end - start_time)
            .count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};