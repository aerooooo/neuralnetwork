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
	Layer	[]NNLayer		// Коллекция слоя
	Weight	[]NNWeight		// Коллекция весов
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
		epoch 		= 1000				// Количество эпох (итераций) обучения
		totalError	float32
	)
	matrix := NNMatrix{
		Bias:	1,						// Вводимые данные
		Ratio:	.5,						// Вводимые данные
		Data:  []float32{6.3, 3.2},		// Вводимые данные
		Input: []float32{1.2, 6.3},		// Вводимые данные
	}
	matrix.Layer = []NNLayer{
		{Size: 0},						// Пока 0,
		{Size: 5},						// Вводимые данные
		{Size: 4},						// Вводимые данные
	}

	// Инициализация нейронных слоёв и весов (матрицы)
	matrix.initMatrix()

	// Обучение нейронной сети за какое-то количество эпох
	for i := 0; i < epoch; i++ {
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

// Функция инициализации матрицы
func (matrix *NNMatrix) initMatrix() {
	matrix.Size				 = len(matrix.Layer)
	index					:= matrix.Size - 1				// Индекс выходного (последнего) слоя нейросети
	matrix.Weight			 = make([]NNWeight, index)
	matrix.Layer[0].Size 	 = len(matrix.Input)
	matrix.Layer[index].Size = len(matrix.Data)
	for i := 0; i < matrix.Size; i++ {
		// Создаем срезы для структуры нейронных слоёв и весов
		matrix.Layer[i].Neuron = make([]float32, matrix.Layer[i].Size)
		if i > 0 {
			matrix.Layer[i].Error = make([]float32, matrix.Layer[i].Size)
		}
		if i < index {
			matrix.Layer[i].Neuron  = append(matrix.Layer[i].Neuron, matrix.Bias)
			matrix.Weight[i].Size   = []int{matrix.Layer[i].Size + 1, matrix.Layer[i + 1].Size}
			matrix.Weight[i].Weight = make([][]float32, matrix.Weight[i].Size[0])
			for j := 0; j < matrix.Weight[i].Size[0]; j++ {
				matrix.Weight[i].Weight[j] = make([]float32, matrix.Weight[i].Size[1])
			}
			// Заполняем все веса случайными числами от -0.5 до 0.5
			matrix.Weight[i].fillWeight()
		}
	}
	copy(matrix.Layer[0].Neuron, matrix.Input) // Входные параметры копируем в слои
}

// Функция заполняет все веса случайными числами от -0.5 до 0.5
func (weight *NNWeight) fillWeight() {
	for i := 0; i < weight.Size[0]; i++ {
		for j := 0; j < weight.Size[1]; j++ {
			weight.Weight[i][j] = rand.Float32() - .5
		}
	}
}

// Функция вычисления значения нейронов в слое
func (matrix *NNMatrix) calcNeuron() {
	for i := 1; i < matrix.Size; i++ {
		n := i - 1
		for j := 0; j < matrix.Layer[i].Size; j++ {
			var sum float32 = 0
			for k, v := range matrix.Layer[n].Neuron {
				sum += v * matrix.Weight[n].Weight[k][j]
			}
			matrix.Layer[i].Neuron[j] = getNeuralActivation(sum, 0)
		}
	}
}

// Функция вычисления ошибки выходного нейрона
func (matrix *NNMatrix) calcOutputError() (total float32) {
	total = 0
	j :=  matrix.Size - 1
	for i, v := range matrix.Layer[j].Neuron {
		matrix.Layer[j].Error[i] = (matrix.Data[i] - v) * getDerivativeActivation(v, 0)
		total += (float32)(math.Pow((float64)(matrix.Layer[j].Error[i]), 2))
	}
	return total
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (matrix *NNMatrix) calcError() {
	for i := matrix.Size - 2; i > 0; i-- {
		for j := 0; j < matrix.Layer[i].Size; j++ {
			var sum float32 = 0
			for k, v := range matrix.Layer[i + 1].Error {
				sum += v * matrix.Weight[i].Weight[j][k]
			}
			matrix.Layer[i].Error[j] = sum * getDerivativeActivation(matrix.Layer[i].Neuron[j], 0)
		}
	}
}

// Функция обновления весов
func (matrix *NNMatrix) updateWeight() {
	for i := 1; i < matrix.Size; i++ {
		n := i - 1
		for j, v := range matrix.Layer[i].Error {
			for k := 0; k < matrix.Layer[n].Size; k++ {
				matrix.Weight[n].Weight[k][j] += matrix.Ratio * v * matrix.Layer[n].Neuron[k] * getDerivativeActivation(matrix.Layer[i].Neuron[j], 0)
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
		case val > 1: return 1 + 0.01 * (val - 1)
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
	t := "Layer"
	n := matrix.Size - 1
	for i := 0; i < matrix.Size; i++ {
		if i == len(matrix.Layer) - 1 {
			t = " Output layer"
		}
		fmt.Println(i, t, "size: ", matrix.Layer[i].Size)
		fmt.Println("Neurons:\t", matrix.Layer[i].Neuron)
		fmt.Println("Errors:\t\t", matrix.Layer[i].Error)
	}
	fmt.Println("Weights:")
	for i := 0; i < n; i++ {
		fmt.Println(matrix.Weight[i].Weight)
	}
	fmt.Println("Total Error:\t", total)
}