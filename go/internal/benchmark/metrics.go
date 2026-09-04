package benchmark

import (
	"github.com/Anupam-Hari/cuml-go/go/internal/capi"
)

type Metrics struct {
	cpu *capi.CPUMonitor
}

func NewMetrics() (*Metrics, error) {

	cpu, err := capi.NewCPUMonitor()
	if err != nil {
		return nil, err
	}

	return &Metrics{
		cpu: cpu,
	}, nil
}

func (m *Metrics) Start() {
	m.cpu.Start()
}

func (m *Metrics) Stop() {
	m.cpu.Stop()
}

func (m *Metrics) Close() {
	if m.cpu != nil {
		m.cpu.Close()
	}
}

func (m *Metrics) CPUAverage() float64 {
	return m.cpu.Average()
}