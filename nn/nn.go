package nn

import (
	"log"
	"math"
	"math/rand"
	"time"
)

const (
	DEFRATE float64 = .3		// Default rate
	MINLOSS float64 = 10e-33	// Минимальная величина средней квадратичной суммы ошибки при достижении которой обучение прекращается принудительно
	MAXITER int     = 10e+05	// Максимальная количество итреаций по достижению которой обучение прекращается принудительно
	MSE		uint8   = 0			// Mean Squared Error
	RMSE	uint8   = 1			// Root Mean Squared Error
	ARCTAN	uint8   = 2			// Arctan
)

// Collection of neural network matrix parameters
type Matrix struct {
	isInit		bool       // Флаг инициализации матрицы
	Size		int        // Количество слоёв в нейросети (Input + Hidden... + Output)
	Index		int        // Индекс выходного (последнего) слоя нейросети
	Mode		uint8      // Идентификатор функции активации
	//ModeActivation
	ModeError	uint8
	Bias		float64    // Нейрон смещения: от 0 до 1
	Rate		float64    // Коэффициент обучения, от 0 до 1
	Limit		float64    // Минимальный (достаточный) уровень средней квадратичной суммы ошибки при обучения
	Hidden		[]int      // Массив количеств нейронов в каждом скрытом слое
	Layer		[]Layer   // Коллекция слоя
	Synapse		[]Synapse // Коллекция весов связей
}

// Collection of neural layer parameters
type Layer struct {
	Size   int       // Количество нейронов в слое
	Neuron []float64 // Значения нейрона
	Error  []float64 // Значение ошибки
}

// Collection of weight parameters
type Synapse struct {
	Size   []int       // Количество связей весов {X, Y}, X - входной (предыдущий) слой, Y - выходной (следующий) слой
	Weight [][]float64 // Значения весов
}

type (
	FloatType float64
	Bias      FloatType
	Rate      FloatType
	Limit     FloatType
)

// Matrix initialization function
func (m *Matrix) InitMatrix(mode uint8, bias, rate, limit FloatType, input, target []float64, hidden []int) {
	m.Mode   = mode
	m.Bias   = bias.Checking()
	m.Rate   = rate.Checking()
	m.Limit  = limit.Checking()
	m.Hidden = hidden
	m.isInit = m.Initializing(input, target)
}

func (f FloatType) Checking() float64 {
	return float64(f)
}

func (b Bias) Checking() float64 {
	switch {
	case b < 0:
		return 0
	case b > 1:
		return 1
	default:
		return float64(b)
	}
}

func (r Rate) Checking() float64 {
	switch {
	case r < 0 || r > 1:
		return DEFRATE
	default:
		return float64(r)
	}
}

func (l Limit) Checking() float64 {
	switch {
	case l < 0:
		return MINLOSS
	default:
		return float64(l)
	}
}

// Matrix initialization function
func (m *Matrix) Initializing(input, target []float64) bool {
	var i, j int
	layer := []int{len(input)}
	if m.Hidden != nil {
		for i, j = range m.Hidden {
			if j > 0 {
				layer = append(layer, j)
			}
		}
	}
	layer     = append(layer, len(target))
	m.Size    = len(layer)
	m.Index   = m.Size - 1
	m.Layer   = make([]Layer, m.Size)
	m.Synapse = make([]Synapse, m.Index)
	for i, j = range layer {
		m.Layer[i].Size = j
	}
	for i = 0; i < m.Size; i++ {
		m.Layer[i].Neuron = make([]float64, m.Layer[i].Size)
		if i > 0 {
			m.Layer[i].Error = make([]float64, m.Layer[i].Size)
		}
		if i < m.Index {
			m.Layer[i].Neuron = append(m.Layer[i].Neuron, m.Bias)
			m.Synapse[i].Size = []int{m.Layer[i].Size + 1, m.Layer[i+1].Size}
			m.Synapse[i].Weight = make([][]float64, m.Synapse[i].Size[0])
			for j = 0; j < m.Synapse[i].Size[0]; j++ {
				m.Synapse[i].Weight[j] = make([]float64, m.Synapse[i].Size[1])
			}
		}
	}
	copy(m.Layer[0].Neuron, input)
	m.FillWeight()

	return true
}

//
func forwardPropagation() {
}

//
func backwardPropagation() {
}

// Training
/*func (m *Matrix) Training(input, target []float64) (loss float64, count int) {
	if !m.isInit {
		m.isInit = m.Initializing(input, target)
	} else {
		copy(m.Layer[0].Neuron, input)
	}
	m.CalcNeuron()
	loss = m.CalcOutputError(target, MSE)
	m.CalcError()
	m.UpdateWeight()

	return loss, 1
}*/

// Training
func (m *Matrix) Training(input, target []float64) (loss float64, count int) {
	if !m.isInit {
		m.isInit = m.Initializing(input, target)
	} else {
		copy(m.Layer[0].Neuron, input)
	}
	count = 1
	for count <= MAXITER {
		m.CalcNeuron()
		if loss = m.CalcOutputError(target, MSE); loss <= m.Limit || loss <= MINLOSS {
			break
		}
		m.CalcError()
		m.UpdateWeight()
		count++
	}
	return loss, count
}

// Querying
func (m *Matrix) Querying(input []float64) []float64 {
	if m.isInit {
		copy(m.Layer[0].Neuron, input)
	} else {
		log.Panicln("an uninitialized neural network")
	}
	m.CalcNeuron()
	return m.Layer[m.Index].Neuron
}

// The function fills all weights with random numbers from -0.5 to 0.5
func (m *Matrix) FillWeight() {
	rand.Seed(time.Now().UTC().UnixNano())
	randWeight := func() float64 {
		r := 0.
		for r == 0 {
			r = rand.Float64() - .5
		}
		return r
	}
	for i := 0; i < m.Index; i++ {
		n := m.Synapse[i].Size[0] - 1
		for j := 0; j < m.Synapse[i].Size[0]; j++ {
			for k := 0; k < m.Synapse[i].Size[1]; k++ {
				if j == n && m.Bias == 0 {
					m.Synapse[i].Weight[j][k] = 0
				} else {
					m.Synapse[i].Weight[j][k] = randWeight()
				}
			}
		}
	}
}

// Function for calculating the values of neurons in a layers
func (m *Matrix) CalcNeuron() {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j := 0; j < m.Layer[i].Size; j++ {
			sum := 0.
			for k, v := range m.Layer[n].Neuron {
				sum += v * m.Synapse[n].Weight[k][j]
			}
			m.Layer[i].Neuron[j] = GetActivation(sum, m.Mode)
		}
	}
}

// Function for calculating the error of the output neuron
func (m *Matrix) CalcOutputError(target []float64, modeError uint8) (loss float64) {
	loss = 0
	for i, v := range m.Layer[m.Index].Neuron {
		m.Layer[m.Index].Error[i] = target[i] - v
		switch modeError {
		default: fallthrough
		case MSE, RMSE:
			loss += math.Pow(m.Layer[m.Index].Error[i], 2)
		case ARCTAN:
			loss += math.Pow(math.Atan(m.Layer[m.Index].Error[i]), 2)
		}
		m.Layer[m.Index].Error[i] *= GetDerivative(v, m.Mode)
	}
	loss /= float64(m.Layer[m.Index].Size)
	switch modeError {
	default: fallthrough
	case MSE, ARCTAN:
		return loss
	case RMSE:
		return math.Sqrt(loss)
	}
}

// Function for calculating the error of neurons in hidden layers
func (m *Matrix) CalcError() {
	for i := m.Index - 1; i > 0; i-- {
		for j := 0; j < m.Layer[i].Size; j++ {
			m.Layer[i].Error[j] = 0.
			for k, v := range m.Layer[i + 1].Error {
				m.Layer[i].Error[j] += v * m.Synapse[i].Weight[j][k]
			}
			m.Layer[i].Error[j] *= GetDerivative(m.Layer[i].Neuron[j], m.Mode)
		}
	}
}

// Weights update function
func (m *Matrix) UpdateWeight() {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j, v := range m.Layer[i].Error {
			for k, p := range m.Layer[n].Neuron {
				m.Synapse[n].Weight[k][j] += v * p * m.Rate
				//if m.Synapse[n].Weight[k][j] == 0 {fmt.Println(m.Synapse[n].Weight[k][j])}
				//if math.Abs(m.Synapse[n].Weight[k][j]) > 1 {fmt.Println(m.Synapse[n].Weight[k][j])}
			}
		}
	}
}
