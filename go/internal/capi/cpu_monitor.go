package capi

/*
#include <sys/resource.h>
*/
import "C"

import (
	"fmt"
	"time"
)

type CPUMonitor struct {
	startCPU  time.Duration
	startWall time.Time
	cpuAvg    float64
	running   bool
}

func NewCPUMonitor() (*CPUMonitor, error) {
	return &CPUMonitor{}, nil
}

func readProcessCPUTime() (time.Duration, error) {
	var usage C.struct_rusage

	if C.getrusage(C.RUSAGE_SELF, &usage) != 0 {
		return 0, fmt.Errorf("getrusage failed")
	}

	user := time.Duration(usage.ru_utime.tv_sec)*time.Second +
		time.Duration(usage.ru_utime.tv_usec)*time.Microsecond

	system := time.Duration(usage.ru_stime.tv_sec)*time.Second +
		time.Duration(usage.ru_stime.tv_usec)*time.Microsecond

	return user + system, nil
}

func (c *CPUMonitor) Start() error {
	if c.running {
		return nil
	}

	cpu, err := readProcessCPUTime()
	if err != nil {
		return err
	}
	c.startCPU = cpu
	c.startWall = time.Now()
	c.cpuAvg = 0
	c.running = true

	return nil
}

func (c *CPUMonitor) Stop() {
	if !c.running {
		return
	}

	endCPU, err := readProcessCPUTime()
	endWall := time.Now()

	if err == nil {
		cpuElapsed := endCPU - c.startCPU
		wallElapsed := endWall.Sub(c.startWall)

		fmt.Printf(
			"CPU elapsed: %v, Wall elapsed: %v, CPU cores: %.2f, CPU %%: %.2f\n",
			cpuElapsed,
			wallElapsed,
			cpuElapsed.Seconds()/wallElapsed.Seconds(),
			(cpuElapsed.Seconds()/wallElapsed.Seconds())*100,
		)

		if wallElapsed > 0 {
			c.cpuAvg =
				(float64(cpuElapsed) / float64(wallElapsed)) * 100.0
		}
	}

	c.running = false
}

func (c *CPUMonitor) Average() float64 {
	return c.cpuAvg
}

func (c *CPUMonitor) Close() {}