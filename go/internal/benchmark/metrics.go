package benchmark

import (
	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
)

type Metrics struct {
	cpu *capi.CPUMonitor
	gpu *capi.GPUMonitor
}

func NewMetrics() (*Metrics, error) {

	cpu, err := capi.NewCPUMonitor()
	if err != nil {
		return nil, err
	}

	gpu, err := capi.NewGPUMonitor()
	if err != nil {
		cpu.Close()
		return nil, err
	}

	return &Metrics{
		cpu: cpu,
		gpu: gpu,
	}, nil
}

func (m *Metrics) Start() {
	m.cpu.Start()
	m.gpu.Start()
}

func (m *Metrics) Stop() {
	m.cpu.Stop()
	m.gpu.Stop()
}

func (m *Metrics) Close() {
	if m.cpu != nil {
		m.cpu.Close()
	}
	if m.gpu != nil {
		m.gpu.Close()
	}
}

func (m *Metrics) CPUAverage() float64 {
	return m.cpu.Average()
}

func (m *Metrics) CPUPeak() float64 {
	return m.cpu.Peak()
}

func (m *Metrics) GPUAverage() float64 {
	return m.gpu.Average()
}

func (m *Metrics) GPUPeak() float64 {
	return m.gpu.Peak()
}

func (m *Metrics) CPUMemoryAverage() float64 {
	return m.cpu.MemoryAverage()
}

func (m *Metrics) CPUMemoryPeak() float64 {
	return m.cpu.MemoryPeak()
}

func (m *Metrics) GPUMemoryAverage() float64 {
	return m.gpu.MemoryAverage()
}

func (m *Metrics) GPUMemoryPeak() float64 {
	return m.gpu.MemoryPeak()
}