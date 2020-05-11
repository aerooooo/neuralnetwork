package nn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const (
	DEFRATE float64 = .3      // Default rate
	MINLOSS float64 = .0001   // Минимальная величина средней квадратичной суммы ошибки при достижении которой обучение прекращается принудительно
	MAXITER int     = 1000000 // Максимальная количество итреаций по достижению которой обучение прекращается принудительно
)

// Collection of neural network matrix parameters
type Matrix struct {
	IsInit bool       // Флаг инициализации матрицы
	Size   int        // Количество слоёв в нейросети (Input + Hidden... + Output)
	Index  int        // Индекс выходного (последнего) слоя нейросети
	Mode   uint8      // Идентификатор функции активации
	Bias   float64    // Нейрон смещения: от 0 до 1
	Rate   float64    // Коэффициент обучения, от 0 до 1
	Limit  float64    // Минимальный (достаточный) уровень средней квадратичной суммы ошибки при обучения
	Hidden []int      // Массив количеств нейронов в каждом скрытом слое
	Layer   []Layer   // Коллекция слоя
	Synapse []Synapse // Коллекция весов связей
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
func (m *Matrix) InitMatrix(mode uint8, bias, rate, limit FloatType, input, data []float64, hidden []int) {
	m.Mode   = mode
	m.Bias   = bias.Checking()
	m.Rate   = rate.Checking()
	m.Limit  = limit.Checking()
	m.Hidden = hidden
	m.IsInit = m.Initializing(input, data)
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
func (m *Matrix) Initializing(input, data []float64) bool {
	var i, j int
	layer := []int{len(input)}
	if m.Hidden != nil {
		for i, j = range m.Hidden {
			if j > 0 {
				layer = append(layer, j)
			}
		}
	}
	layer     = append(layer, len(data))
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

// Training
func (m *Matrix) Training(input, data []float64) (count int, loss float64) {
	if !m.IsInit {
		m.IsInit = m.Initializing(input, data)
	} else {
		copy(m.Layer[0].Neuron, input)
	}
	count = 1
	for count <= 10/*MAXITER*/ {
		m.CalcNeuron()
		if loss = m.CalcOutputError(data); loss <= m.Limit || loss <= MINLOSS {
			//fmt.Println("+++", loss," ",count)
			break
		}
		m.CalcError()
		m.UpdateWeight()
		count++

	}
	return count, loss
}

// The function fills all weights with random numbers from -0.5 to 0.5
func (m *Matrix) FillWeight() {
	rand.Seed(time.Now().UTC().UnixNano())
	for i := 0; i < m.Index; i++ {
		n := m.Synapse[i].Size[0] - 1
		for j := 0; j < m.Synapse[i].Size[0]; j++ {
			for k := 0; k < m.Synapse[i].Size[1]; k++ {
				if j == n && m.Bias == 0 {
					m.Synapse[i].Weight[j][k] = 0
				} else {
					m.Synapse[i].Weight[j][k] = rand.Float64() - .5
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
			/*go m.GetNeuron(i, j, k)*/
			sum := 0.
			for k, v := range m.Layer[n].Neuron {
				sum += v * m.Synapse[n].Weight[k][j]
			}
			m.Layer[i].Neuron[j] = GetActivation(sum, m.Mode)
			fmt.Println(sum,m.Layer[i].Neuron[j])
		}
	}
}

/*func (m *Matrix) GetNeuron(x, y, z int) {
	var sum float32 = 0
	for i, v := range m.Layer[z].Neuron {
		sum += v * m.Synapse[z].Weight[i][y]
	}
	m.Layer[x].Neuron[y] = GetActivation(sum, m.Mode)
}*/

// Function for calculating the error of the output neuron
func (m *Matrix) CalcOutputError(data []float64) (loss float64) {
	loss = 0
	for i, v := range m.Layer[m.Index].Neuron {
		m.Layer[m.Index].Error[i] = (data[i] - v) * GetDerivative(v, m.Mode)
		loss += math.Pow(m.Layer[m.Index].Error[i], 2)
	}
	return loss / float64(m.Layer[m.Index].Size)
}

// Function for calculating the error of neurons in hidden layers
func (m *Matrix) CalcError() {
	for i := m.Size - 2; i > 0; i-- {
		for j := 0; j < m.Layer[i].Size; j++ {
			sum := 0.
			for k, v := range m.Layer[i+1].Error {
				sum += v * m.Synapse[i].Weight[j][k]
			}
			m.Layer[i].Error[j] = sum * GetDerivative(m.Layer[i].Neuron[j], m.Mode)
		}
	}
}

// Weights update function
func (m *Matrix) UpdateWeight() {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j, v := range m.Layer[i].Error {
			for k, p := range m.Layer[n].Neuron {
				m.Synapse[n].Weight[k][j] += m.Rate * v * p /** GetDerivative(m.Layer[i].Neuron[j], m.Mode)*/
			}
		}
	}
}
