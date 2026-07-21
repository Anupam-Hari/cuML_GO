#include "gpu_monitor.h"

#include <chrono>
#include <nvml.h>

GpuMonitor::GpuMonitor()
    : running_(false),
      sum_(0.0f),
      peak_(0.0f),
      samples_(0)
{
}

GpuMonitor::~GpuMonitor()
{
    stop();
}

bool GpuMonitor::start()
{
    if (running_)
        return true;

    if (nvmlInit() != NVML_SUCCESS)
        return false;

    if (nvmlDeviceGetHandleByIndex(0, &device_) != NVML_SUCCESS)
    {
        nvmlShutdown();
        return false;
    }

    sum_ = 0.0f;
    peak_ = 0.0f;
    samples_ = 0;

    running_ = true;

    worker_ = std::thread(&GpuMonitor::monitor_loop, this);

    return true;
}

void GpuMonitor::stop()
{
    if (!running_)
        return;

    running_ = false;

    if (worker_.joinable())
        worker_.join();

    nvmlShutdown();
}

void GpuMonitor::monitor_loop()
{
    while (running_)
    {
        nvmlUtilization_t util;

        if (nvmlDeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            float value = static_cast<float>(util.gpu);

            sum_ += value;

            samples_++;

            if (value > peak_)
                peak_ = value;
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(100));
    }
}

float GpuMonitor::average() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (samples_ == 0)
        return 0.0f;

    return sum_ / samples_;
}

float GpuMonitor::peak() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    return peak_;
}

extern "C"
{

GpuMonitorHandle* gpu_monitor_create()
{
    return reinterpret_cast<GpuMonitorHandle*>(new GpuMonitor());
}

int gpu_monitor_start(GpuMonitorHandle* monitor)
{
    auto* m = reinterpret_cast<GpuMonitor*>(monitor);

    return m->start() ? 0 : -1;
}

void gpu_monitor_stop(GpuMonitorHandle* monitor)
{
    auto* m = reinterpret_cast<GpuMonitor*>(monitor);

    m->stop();
}

float gpu_monitor_average(GpuMonitorHandle* monitor)
{
    auto* m = reinterpret_cast<GpuMonitor*>(monitor);

    return m->average();
}

float gpu_monitor_peak(GpuMonitorHandle* monitor)
{
    auto* m = reinterpret_cast<GpuMonitor*>(monitor);

    return m->peak();
}

void gpu_monitor_destroy(GpuMonitorHandle* monitor)
{
    auto* m = reinterpret_cast<GpuMonitor*>(monitor);

    delete m;
}

}