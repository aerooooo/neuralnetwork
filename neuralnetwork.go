package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Коллекция параметров нейроных слоёв
type NeuralLayer struct {
	Node   []float32
	Error  []float32
	Weight [][]float32
}

func main() {
	var i, j int
	//bias		:= 1								// Смещение
	inputSet	:= []float32{1.2, 6.3}				// Входные параметры
	dataSet		:= []float32{6.3, 1.2}				// Обучающий набор с которым будет сравниваться выходной слой
	numNeuron	:= []int{5, 4, len(dataSet)}		// Количество нейронов для каждого слоя, исключая входной слой
	numLayer	:= len(numNeuron)					// Количество скрытых слоёв и выходного слоя
	indLayer	:= numLayer - 1						// Индекс последнего слоя нейросети
	layer 		:= make([]NeuralLayer, numLayer)	// Создаём срез нейронных слоёв

	// Первая проходка по нейронным слоям
	for i = 0; i < numLayer; i++ {
		// Создаем срезы для структуры нейронных слоёв
		layer[i].Node   = make([]float32, numNeuron[i])
		layer[i].Error  = make([]float32, numNeuron[i])
		layer[i].Weight = make([][]float32, numNeuron[i])
		for j = 0; j < numNeuron[i]; j++ {
			if i > 0 {
				layer[i].Weight[j] = make([]float32, numNeuron[i - 1])
			} else {
				layer[0].Weight[j] = make([]float32, len(inputSet))
			}
		}

		// Заполняем все веса случайными числами от 0.0 до 1.0
		layer[i].fillWeight()

		// Считаем значения нейронов в слое
		if i > 0 {
			layer[i].calcNeuron(&layer[i - 1].Node)
		} else {
			layer[0].calcNeuron(&inputSet)
		}
	}

	// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
	for i = 0; i < numNeuron[indLayer]; i++ {
		layer[indLayer].Error[i] = dataSet[i] - layer[indLayer].Node[i]
	}

	// Вычисляем ошибки нейронов в скрытых слоях
	for i = indLayer; i > 0; i-- {
		layer[i - 1].Error = layer[i].calcError()
	}

	// Обновление весов

	// Вывод значений нейросети
	printNeuralLayer(&layer)
}

// Функция заполняет все веса случайными числами от 0.0 до 1.0
func (layer *NeuralLayer)  fillWeight() {
	l := *layer
	for i := range l.Weight {
		for j := range l.Weight[i] {
			l.Weight[i][j] = rand.Float32()
		}
	}
}

// Функция вычисления значения нейронов в слое
func (layer *NeuralLayer) calcNeuron(node *[]float32) {
	l := *layer
	n := *node
	for i := range l.Node {
		var sum float32 = 0
		for j := range l.Weight[i] {
			sum += n[j] * l.Weight[i][j]
		}
		l.Node[i] = getNeuralActivation(sum, 0)
	}
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (layer *NeuralLayer) calcError() []float32 {
	l := *layer
	n := len(l.Weight[0])
	e := make([]float32, n)
	for i := 0; i < n; i++ {
		var sum float32 = 0
		for j := range l.Error {
			sum += l.Error[j] * l.Weight[j][i]
		}
		e[i] = sum
	}
	return e
}

// Функция активации нейрона
func getNeuralActivation(sum float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return (float32)(1 / (1 + math.Pow(math.E, (float64)(-sum)))) // Sigmoid
	case 1: // Leaky ReLu
		switch {
		case sum < 0: return 0.01 * sum
		case sum > 1: return 1 + 0.01*(sum - 1)
		default:	  return sum
		}
	}
}

// Функция вывода результатов нейросети
func printNeuralLayer(layer *[]NeuralLayer) {
	l := *layer
	t := ""
	for i := range l {
		if i == len(l) - 1 {
			t = " Output layer"
		}
		fmt.Println(i, t)
		fmt.Println("Weights:\t",  l[i].Weight)
		fmt.Println("Neurons:\t",  l[i].Node)
		fmt.Println("Errors:\t\t", l[i].Error)
	}
}