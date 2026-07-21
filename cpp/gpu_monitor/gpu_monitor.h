#pragma once

typedef struct nvmlDevice_st* nvmlDevice_t;

#ifdef __cplusplus
#include <atomic>
#include <mutex>
#include <thread>
#endif

#ifdef __cplusplus

class GpuMonitor
{
public:
    GpuMonitor();
    ~GpuMonitor();

    bool start();
    void stop();

    float average() const;
    float peak() const;

private:
    void monitor_loop();

private:
    nvmlDevice_t device_;

    std::thread worker_;
    std::atomic<bool> running_;

    mutable std::mutex mutex_;

    float sum_;
    float peak_;
    int samples_;
};

extern "C" {
#endif

typedef struct GpuMonitor GpuMonitorHandle;

GpuMonitorHandle* gpu_monitor_create();

int gpu_monitor_start(GpuMonitorHandle* monitor);

void gpu_monitor_stop(GpuMonitorHandle* monitor);

float gpu_monitor_average(GpuMonitorHandle* monitor);

float gpu_monitor_peak(GpuMonitorHandle* monitor);

void gpu_monitor_destroy(GpuMonitorHandle* monitor);

#ifdef __cplusplus
}
#endif