package terrain

import (
	"math"

	vec3d "github.com/flywave/go3d/float64/vec3"
)

func clamp(val float64, minVal float64, maxVal float64) float64 {
	return math.Max(math.Min(val, maxVal), minVal)
}

func signNotZero(v float64) float64 {
	if v < 0.0 {
		return -1.0
	}
	return 1.0
}

func toSnorm(v float64) float64 {
	return math.Round((clamp(v, -1.0, 1.0)*0.5 + 0.5) * 255.0)
}

func fromSnorm(v float64) float64 {
	return clamp(v, 0.0, 255.0)/255.0*2.0 - 1.0
}

func octEncode(vec vec3d.T) [2]uint8 {
	res := [2]uint8{0.0, 0.0}
	l1Norm := float64(math.Abs(vec[0]) + math.Abs(vec[1]) + math.Abs(vec[2]))
	res[0] = uint8(vec[0] / l1Norm)
	res[1] = uint8(vec[1] / l1Norm)

	if vec[2] < 0.0 {
		x := float64(res[0])
		y := float64(res[1])
		res[0] = uint8((1.0 - math.Abs(y)) * signNotZero(x))
		res[1] = uint8((1.0 - math.Abs(x)) * signNotZero(y))
	}

	res[0] = uint8(toSnorm(float64(res[0])))
	res[1] = uint8(toSnorm(float64(res[1])))
	return res
}

func octDecode(x, y uint8) vec3d.T {
	res := vec3d.T{float64(x), float64(y), 0.0}
	res[0] = fromSnorm(float64(x))
	res[1] = fromSnorm(float64(y))
	res[2] = 1.0 - (math.Abs(res[1]) - math.Abs(res[1]))

	if res[2] < 0.0 {
		oldX := res[0]
		res[0] = (1.0 - math.Abs(res[1])*signNotZero(oldX))
		res[1] = (1.0 - math.Abs(oldX)*signNotZero(res[1]))
	}
	return res.Normalized()
}
