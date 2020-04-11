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

	// Создаем срезы для структуры нейронных слоёв
	for i = 0; i < numLayer; i++ {
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
	}

	// Заполняем все веса случайными числами от 0.0 до 1.0
	fmt.Println("Weights:")
	fillWeight(&layer)

	// Считаем значения нейронов в слое
	fmt.Println("Neurons:")
	for i = 0; i < numLayer; i++ {
		if i > 0 {
			layer[i].Node = layer[i].calcNeuron(&layer[i - 1].Node)
		} else {
			layer[0].Node = layer[0].calcNeuron(&inputSet)
		}
		fmt.Println(i," ",layer[i].Node)
	}

	// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
	for i = 0; i < numNeuron[indLayer]; i++ {
		layer[indLayer].Error[i] = dataSet[i] - layer[indLayer].Node[i]
	}
	fmt.Println("Errors:")
	fmt.Println(indLayer," ",layer[indLayer].Error)

	// Вычисляем ошибки нейронов в скрытых слоях
	for i = indLayer; i > 0; i-- {
		layer[i - 1].Error = layer[i].calcError()
		fmt.Println(i," ",layer[i - 1].Error)
	}

	fmt.Println(layer)
}

// Функция заполняет все веса случайными числами от 0.0 до 1.0
func fillWeight(layer *[]NeuralLayer) {
	l := *layer
	for i := range l {
		for j := range l[i].Weight {
			for k := range l[i].Weight[j] {
				l[i].Weight[j][k] = rand.Float32()
			}
		}
		fmt.Println(i," ",l[i].Weight)
	}
}

// Функция вычисления значения нейронов в слое
func (layer *NeuralLayer) calcNeuron(node *[]float32) []float32 {
	l := *layer
	n := *node
	for i := range l.Node {
		var sum float32 = 0
		for j := range l.Weight[i] {
			sum += n[j] * l.Weight[i][j]
		}
		l.Node[i] = getNeuralActivation(sum, 0)
	}
	return l.Node
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (layer *NeuralLayer) calcError() []float32 {
	l := *layer
	e := make([]float32, len(l.Weight[0]))
	for i := range l.Weight[0] {
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