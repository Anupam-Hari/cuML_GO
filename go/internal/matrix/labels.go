package matrix

import "fmt"

func NumClasses(labels []int) (int, error) {
	if len(labels) == 0 {
		return 0, fmt.Errorf("empty labels")
	}

	classes := make(map[int]struct{})

	for _, l := range labels {
		classes[l] = struct{}{}
	}

	return len(classes), nil
}

func ToCInt(labels []int) []int32 {
	out := make([]int32, len(labels))

	for i, v := range labels {
		out[i] = int32(v)
	}

	return out
}

func FromCInt32(values []int32) []int {
	out := make([]int, len(values))

	for i, v := range values {
		out[i] = int(v)
	}

	return out
}
