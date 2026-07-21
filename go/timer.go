package main

import "time"

type Timer struct {
	start time.Time
}

func (t *Timer) Start() {
	t.start = time.Now()
}

func (t *Timer) Stop() float64 {
	return float64(time.Since(t.start).Microseconds()) / 1000.0
}