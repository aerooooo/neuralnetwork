package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Коллекция параметров нейроных слоёв
type NeuralLayer struct {
	Size   int
	Node   []float32
	Error  []float32
	Weight [][]float32
}

func main() {
	var (
		i, j int
		ratioLearn float32 = .5						// Коэффициент обучения, от 0 до 1
	)
	//biasNeuron	:= 1								// Нейрон смещения
	dataSet		:= []float32{6.3, 1.2}				// Обучающий набор с которым будет сравниваться выходной слой
	inputSet	:= []float32{1.2, 6.3}				// Входные параметры (слой)
	numNeuron	:= []int{5, 4, len(dataSet)}		// Количество нейронов для каждого слоя, исключая входной слой
	numInput	:= len(inputSet)					// Количество входных нейронов
	numLayer	:= len(numNeuron)					// Количество скрытых слоёв и выходного слоя
	indOutput	:= numLayer - 1						// Индекс выходного (последнего) слоя нейросети
	layer 		:= make([]NeuralLayer, numLayer)	// Создаём срез нейронных слоёв

	// Инициализация нейронных слоёв
	for i = 0; i < numLayer; i++ {
		// Создаем срезы для структуры нейронных слоёв
		layer[i].Size   = numNeuron[i]
		layer[i].Node   = make([]float32,   numNeuron[i])
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

	// Вычисляем значения нейронов в слое
	layer[0].calcNeuron(&inputSet)
	for i = 1; i < numLayer; i++ {
		layer[i].calcNeuron(&layer[i - 1].Node)
	}

	// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
	var totalError float32 = 0
	for i = 0; i < numNeuron[indOutput]; i++ {
		layer[indOutput].Error[i] = (dataSet[i] - layer[indOutput].Node[i]) * getDerivativeActivation(layer[indOutput].Node[i], 0)
		totalError += (float32)(math.Pow((float64)(layer[indOutput].Error[i]), 2))
	}

	// Вычисляем ошибки нейронов в скрытых слоях
	for i = indOutput; i > 0; i-- {
		layer[i - 1].calcError(&layer[i])
	}

	// Обновление весов
	layer[0].updateWeight(ratioLearn, &inputSet)
	for i = 1; i < numLayer; i++ {
		layer[i].updateWeight(ratioLearn, &layer[i - 1].Node)
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

//
func d(iter uint) float32 {

	return 0
}

// Функция вычисления значения нейронов в слое
func (layer *NeuralLayer) calcNeuron(node *[]float32) {
	l := *layer
	n := *node
	for i := 0; i < l.Size; i++ {
		var sum float32 = 0
		for j, w:= range l.Weight[i] {
			sum += n[j] * w
		}
		l.Node[i] = getNeuralActivation(sum, 0)
	}
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (layer *NeuralLayer) calcError(node *NeuralLayer) {
	l := *layer
	n := *node
	for i, v := range l.Node {
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
	for i, v := range l.Error {
		for j, w := range n {
			l.Weight[i][j] += ratio * v * w * getDerivativeActivation(l.Node[i], 0)
		}
	}
}

// Функция активации нейрона
func getNeuralActivation(v float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return (float32)(1 / (1 + math.Pow(math.E, (float64)(-v)))) // Sigmoid
	case 1: // Leaky ReLu
		switch {
		case v < 0: return 0.01 * v
		case v > 1: return 1 + 0.01*(v - 1)
		default:	return v
		}
	}
}

// Функция производной активации
func getDerivativeActivation(v float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return v * (1 - v)
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
		fmt.Println("Neurons:\t",	l[i].Node)
		fmt.Println("Errors:\t\t",	l[i].Error)
	}
	fmt.Println("Total Error:\t",  err)
}