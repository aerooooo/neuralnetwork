package nn

import (
	"math"
)

const (
	LINEAR    uint8 = 0 // Linear/Identity (линейная/тождественная)
	SIGMOID   uint8 = 1 // Logistic, a.k.a. sigmoid or soft step (логистическая, сигмоида или гладкая ступенька)
	TANH      uint8 = 2 // TanH - hyperbolic (гиперболический тангенс)
	RELU      uint8 = 3 // ReLu - rectified linear unit (линейный выпрямитель)
	LEAKYRELU uint8 = 4 // Leaky ReLu - leaky rectified linear unit (линейный выпрямитель с «утечкой»)
)

// Activation function
func GetActivation(value float64, mode uint8) float64 {
	switch mode {
	default:
		fallthrough
	case LINEAR:
		return value
	case SIGMOID:
		return 1 / (1 + math.Exp(-value))
	case TANH:
		value = math.Exp(2 * value)
		if math.IsInf(value, 1) {
			return 1
		}
		return (value - 1) / (value + 1)
	case RELU:
		switch {
		case value < 0:
			return 0
		default:
			return value
		}
	case LEAKYRELU:
		switch {
		case value < 0:
			return .01 * value
		default:
			return value
		}
	}
}

// Derivative activation function
func GetDerivative(value float64, mode uint8) float64 {
	switch mode {
	default:
		fallthrough
	case LINEAR:
		return 1
	case SIGMOID:
		return value * (1 - value)
	case TANH:
		return 1 - math.Pow(value, 2)
	case RELU:
		switch {
		case value < 0:
			return 0
		default:
			return 1
		}
	case LEAKYRELU:
		switch {
		case value < 0:
			return .01
		default:
			return 1
		}
	}
}
