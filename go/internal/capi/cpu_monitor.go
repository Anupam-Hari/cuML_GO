package capi

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type CPUMonitor struct {
	mu      sync.Mutex
	running bool
	done    chan struct{}

	sum   float64
	count int
	peak  float64
}

func NewCPUMonitor() (*CPUMonitor, error) {
	return &CPUMonitor{}, nil
}

func readProcStat() (uint64, error) {
	f, err := os.Open("/proc/stat")
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	if !scanner.Scan() {
		return 0, fmt.Errorf("failed to read /proc/stat")
	}

	fields := strings.Fields(scanner.Text())
	if len(fields) < 8 {
		return 0, fmt.Errorf("invalid /proc/stat")
	}

	var total uint64
	for _, s := range fields[1:] {
		v, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return 0, err
		}
		total += v
	}

	return total, nil
}

func readProcSelfStat() (uint64, error) {
	data, err := os.ReadFile("/proc/self/stat")
	if err != nil {
		return 0, err
	}

	fields := strings.Fields(string(data))
	if len(fields) < 17 {
		return 0, fmt.Errorf("invalid /proc/self/stat")
	}

	utime, err := strconv.ParseUint(fields[13], 10, 64)
	if err != nil {
		return 0, err
	}

	stime, err := strconv.ParseUint(fields[14], 10, 64)
	if err != nil {
		return 0, err
	}

	return utime + stime, nil
}

func (c *CPUMonitor) Start() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.running {
		return nil
	}

	c.running = true
	c.done = make(chan struct{})

	c.sum = 0
	c.count = 0
	c.peak = 0

	go func() {
		prevProc, err := readProcSelfStat()
		if err != nil {
			return
		}

		prevTotal, err := readProcStat()
		if err != nil {
			return
		}

		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()

		ncpu := float64(runtime.NumCPU())

		for {
			select {
			case <-ticker.C:

				curProc, err1 := readProcSelfStat()
				curTotal, err2 := readProcStat()

				if err1 != nil || err2 != nil {
					continue
				}

				dProc := curProc - prevProc
				dTotal := curTotal - prevTotal

				prevProc = curProc
				prevTotal = curTotal

				if dTotal == 0 {
					continue
				}

				usage := (float64(dProc) / float64(dTotal)) * ncpu * 100.0

				c.mu.Lock()

				c.sum += usage
				c.count++

				if usage > c.peak {
					c.peak = usage
				}

				c.mu.Unlock()

			case <-c.done:
				return
			}
		}
	}()

	return nil
}

func (c *CPUMonitor) Stop() {
	c.mu.Lock()

	if !c.running {
		c.mu.Unlock()
		return
	}

	close(c.done)
	c.running = false

	c.mu.Unlock()
}

func (c *CPUMonitor) Average() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.count == 0 {
		return 0
	}

	return c.sum / float64(c.count)
}

func (c *CPUMonitor) Peak() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.peak
}

func (c *CPUMonitor) Close() {}