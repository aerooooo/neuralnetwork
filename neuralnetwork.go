package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Коллекция параметров нейроных слоёв
type NeuralLayer struct {
	Size	int
	Neuron	[]float32
	Error	[]float32
	Weight	[][]float32
}

func main() {
	var (
		i, j, k int
		totalError float32
		ratioLearn float32 = .5						// Коэффициент обучения, от 0 до 1
		biasNeuron float32 = 1						// Нейрон смещения
	)
	i = (int)(biasNeuron)
	setData		:= []float32{6.3, 3.2}				// Обучающий набор с которым будет сравниваться выходной слой
	setInput	:= []float32{1.2, 6.3, biasNeuron}	// Входные параметры (слой)
	numNeuron	:= []int{5 + i, 4 + i, len(setData)}// Количество нейронов для каждого слоя, исключая входной слой
	numInput	:= len(setInput)					// Количество входных нейронов
	numLayer	:= len(numNeuron)					// Количество скрытых слоёв и выходного слоя
	indOutput	:= numLayer - 1						// Индекс выходного (последнего) слоя нейросети
	layer 		:= make([]NeuralLayer, numLayer)	// Создаём срез нейронных слоёв
	epoch       := 1000								// Количество эпох (итераций) обучения

	// Инициализация нейронных слоёв
	for i = 0; i < numLayer; i++ {
		// Создаем срезы для структуры нейронных слоёв
		layer[i].Size   = numNeuron[i]
		layer[i].Neuron = make([]float32,   numNeuron[i])
		layer[i].Error  = make([]float32,   numNeuron[i])
		layer[i].Weight = make([][]float32, numNeuron[i])
		for j = 0; j < numNeuron[i]; j++ {
			if i > 0 {
				layer[i].Weight[j] = make([]float32, numNeuron[i - 1])
			} else {
				layer[i].Weight[j] = make([]float32, numInput)
			}
		}

		// Заполняем все веса случайными числами от -0.5 до 0.5
		layer[i].fillWeight()
	}

	//
	for k = 0; k < epoch; k++ {
		// Вычисляем значения нейронов в слое
		layer[0].calcNeuron(&setInput)
		for i = 1; i < numLayer; i++ {
			layer[i].calcNeuron(&layer[i-1].Neuron)
		}

		// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
		totalError = 0
		for i = 0; i < numNeuron[indOutput]; i++ {
			layer[indOutput].Error[i] = (setData[i] - layer[indOutput].Neuron[i]) * getDerivativeActivation(layer[indOutput].Neuron[i], 0)
			totalError += (float32)(math.Pow((float64)(layer[indOutput].Error[i]), 2))
		}
		//fmt.Println(k, " ", totalError)

		// Вычисляем ошибки нейронов в скрытых слоях
		for i = indOutput; i > 0; i-- {
			layer[i-1].calcError(&layer[i])
		}

		// Обновление весов
		layer[0].updateWeight(ratioLearn, &setInput)
		for i = 1; i < numLayer; i++ {
			layer[i].updateWeight(ratioLearn, &layer[i-1].Neuron)
		}
	}

	// Вывод значений нейросети
	printLayer(totalError, &layer)
}

// Функция заполняет все веса случайными числами от -0.5 до 0.5
func (layer *NeuralLayer) fillWeight() {
	l := *layer
	for i := 0; i < l.Size; i++ {
		for j := range l.Weight[i] {
			l.Weight[i][j] = rand.Float32() - .5
		}
	}
}

// Функция вычисления значения нейронов в слое
func (layer *NeuralLayer) calcNeuron(node *[]float32) {
	l := *layer
	n := *node
	for i := 0; i < l.Size; i++ {
		var sum float32 = 0
		for j, y:= range l.Weight[i] {
			sum += n[j] * y
		}
		l.Neuron[i] = getNeuralActivation(sum, 0)
	}
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (layer *NeuralLayer) calcError(node *NeuralLayer) {
	l := *layer
	n := *node
	for i, v := range l.Neuron {
		var sum float32 = 0
		for j, w := range n.Error {
			sum += w * n.Weight[j][i]
		}
		l.Error[i] = sum * getDerivativeActivation(v, 0)
	}
}

// Функция обновления весов
func (layer *NeuralLayer) updateWeight(ratio float32, node *[]float32) {
	l := *layer
	n := *node
	for i, e := range l.Error {
		for j, v := range n {
			l.Weight[i][j] += ratio * e * v * getDerivativeActivation(l.Neuron[i], 0)
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
func printLayer(err float32, layer *[]NeuralLayer) {
	l := *layer
	t := "Layer"
	for i := range l {
		if i == len(l) - 1 {
			t = " Output layer"
		}
		fmt.Println(i, t, "size: ",		l[i].Size)
		fmt.Println("Weights:\t",	l[i].Weight)
		fmt.Println("Neurons:\t",	l[i].Neuron)
		fmt.Println("Errors:\t\t",	l[i].Error)
	}
	fmt.Println("Total Error:\t",  err)
}