package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Коллекция параметров матрицы
type NNMatrix struct {
	Size	int				// Количество слоёв в нейросети (Input + Hidden + Output)
	Bias	float32			// Нейрон смещения
	Ratio 	float32			// Коэффициент обучения, от 0 до 1
	Data	[]float32		// Обучающий набор с которым будет сравниваться выходной слой
	Input	[]float32		// Входные параметры
	Layer	*[]NNLayer		// Коллекция полей слоя
	Weight	*[]NNWeight		// Коллекция массива весов
}

// Коллекция параметров нейронного слоя
type NNLayer struct {
	Size	int				// Количество нейронов в слое
	Neuron	[]float32		// Значения нейрона
	Error	[]float32		// Значение ошибки

}

// Коллекция параметров весов
type NNWeight struct {
	Size	[]int			// Количество связей весов {X, Y}, X - входной (предыдущий) слой, Y - выходной (следующий) слой
	Weight	[][]float32		// Значения весов
}

func main() {
	var (
		i, j int
		totalError float32
	)
	matrix := NNMatrix{
		Bias:	1,
		Ratio:	.5,
		Data:  []float32{6.3, 3.2},
		Input: []float32{1.2, 6.3},
	}
	layer := []NNLayer{
		{Size: len(matrix.Input)},
		{Size: 5},
		{Size: 4},
		{Size: len(matrix.Data)},
	}
	matrix.Size = len(layer)
	epoch   := 1000						// Количество эпох (итераций) обучения
	index	:= matrix.Size - 1			// Индекс выходного (последнего) слоя нейросети
	weight	:= make([]NNWeight, index)

	// Инициализация нейронных слоёв и весов
	for i = 0; i < matrix.Size; i++ {
		// Создаем срезы для структуры нейронных слоёв и весов
		layer[i].Neuron = make([]float32, layer[i].Size)
		layer[i].Error  = make([]float32, layer[i].Size)
		if i < index {
			weight[i].Size   = []int{layer[i].Size, layer[i + 1].Size}
			weight[i].Weight = make([][]float32, weight[i].Size[0])
			for j = 0; j < weight[i].Size[0]; j++ {
				weight[i].Weight[j] = make([]float32, weight[i].Size[1])
			}
			// Заполняем все веса случайными числами от -0.5 до 0.5
			weight[i].fillWeight()
		}
	}
	copy(layer[0].Neuron, matrix.Input) // Входные параметры
	matrix.Layer  = &layer
	matrix.Weight = &weight

	// Обучение нейронной сети за какое-то количество эпох
	for i = 0; i < epoch; i++ {
		// Вычисляем значения нейронов в слое
		matrix.calcNeuron()

		// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
		totalError = matrix.calcOutputError()

		// Вычисляем ошибки нейронов в скрытых слоях
		matrix.calcError()

		// Обновление весов
		matrix.updateWeight()
	}

	// Вывод значений нейросети
	matrix.printMatrix(totalError)
}

// Функция заполняет все веса случайными числами от -0.5 до 0.5
func (weight *NNWeight) fillWeight() {
	w := *weight
	for i := 0; i < w.Size[0]; i++ {
		for j := 0; j < w.Size[1]; j++ {
			w.Weight[i][j] = rand.Float32() - .5
		}
	}
}

// Функция вычисления значения нейронов в слое
func (matrix *NNMatrix) calcNeuron() {
	m := *matrix
	l := *m.Layer
	w := *m.Weight
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j := 0; j < l[i].Size; j++ {
			var sum float32 = 0
			for k, z := range l[n].Neuron {
				sum += z * w[n].Weight[k][j]
			}
			l[i].Neuron[j] = getNeuralActivation(sum, 0)
		}
	}
}

// Функция вычисления ошибки выходного нейрона
func (matrix *NNMatrix) calcOutputError() (total float32) {
	total = 0
	m := *matrix
	l := *m.Layer
	j :=  m.Size - 1
	for i, x := range l[j].Neuron {
		l[j].Error[i] = (m.Data[i] - x) * getDerivativeActivation(x, 0)
		total += (float32)(math.Pow((float64)(l[j].Error[i]), 2))
	}
	return total
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (matrix *NNMatrix) calcError() {
	m := *matrix
	l := *m.Layer
	w := *m.Weight
	for i := m.Size - 2; i > 0; i-- {
		n := i + 1
		for j, y := range l[i].Neuron {
			var sum float32 = 0
			for k, z := range l[n].Error {
				sum += z * w[i].Weight[j][k]
			}
			l[i].Error[j] = sum * getDerivativeActivation(y, 0)
		}
	}
}

// Функция обновления весов
func (matrix *NNMatrix) updateWeight() {
	m := *matrix
	l := *m.Layer
	w := *m.Weight
	for i := 1; i < m.Size; i++ {
		for j, y := range l[i].Error {
			for k, z := range l[i - 1].Neuron {
				w[i - 1].Weight[k][j] += m.Ratio * y * z * getDerivativeActivation(l[i].Neuron[j], 0)
			}
		}
	}
}

// Функция активации нейрона
func getNeuralActivation(val float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return (float32)(1 / (1 + math.Pow(math.E, (float64)(-val)))) // Sigmoid
	case 1: // Leaky ReLu
		switch {
		case val < 0: return 0.01 * val
		case val > 1: return 1 + 0.01*(val - 1)
		default:	  return val
		}
	case 2: return (float32)(2 / (1 + math.Pow(math.E, (float64)(-2 * val))) - 1) // Tanh - гиперболический тангенс
	}
}

// Функция производной активации
func getDerivativeActivation(val float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return val * (1 - val)
	case 1: return 1
	}
}

// Функция вывода результатов нейросети
func (matrix *NNMatrix) printMatrix(total float32) {
	m := *matrix
	l := *m.Layer
	w := *m.Weight
	t := "Layer"
	n := m.Size - 1
	for i := 0; i < m.Size; i++ {
		if i == len(l) - 1 {
			t = " Output layer"
		}
		fmt.Println(i, t, "size: ",		l[i].Size)
		fmt.Println("Neurons:\t",	l[i].Neuron)
		fmt.Println("Errors:\t\t",	l[i].Error)
		if i < n {
			fmt.Println("Weights:\t", w[i].Weight)
		}
	}
	fmt.Println("Total Error:\t",  total)
}