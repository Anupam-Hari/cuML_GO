package dataset

import (
	"fmt"
	"strconv"

	"github.com/sjwhitworth/golearn/base"
)

func ToGoLearnInstances(
	X [][]float32,
	y []int,
) (*base.DenseInstances, error) {

	if len(X) == 0 {
		return nil, fmt.Errorf("empty dataset")
	}

	if len(X) != len(y) {
		return nil, fmt.Errorf("number of samples and labels differ")
	}

	nRows := len(X)
	nCols := len(X[0])

	inst := base.NewDenseInstances()

	attrs := make([]base.Attribute, nCols)
	attrSpecs := make([]base.AttributeSpec, nCols)

	for i := 0; i < nCols; i++ {
		attrs[i] = base.NewFloatAttribute(
			fmt.Sprintf("feature_%d", i),
		)

		attrSpecs[i] = inst.AddAttribute(attrs[i])
	}

	classAttr := base.NewCategoricalAttribute("class")
	classSpec := inst.AddClassAttribute(classAttr)

	inst.Extend(nRows)

	for r := 0; r < nRows; r++ {

		if len(X[r]) != nCols {
			return nil, fmt.Errorf("row %d has inconsistent column count", r)
		}

		for c := 0; c < nCols; c++ {
			inst.Set(attrSpecs[c], r, attrs[c].GetSysValFromFloat(float64(X[r][c])))
		}

		inst.Set(
			classSpec,
			r,
			classAttr.GetSysValFromString(
				strconv.Itoa(y[r]),
			),
		)
	}

	return inst, nil
}