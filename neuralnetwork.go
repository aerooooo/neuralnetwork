package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Коллекция параметров нейроных слоёв
type neuralLayer struct {
	Neuron	[]float64
	Error	[]float64
	Weight	[][]float64
}

func main() {
	numHiddenLayer	:= 2									// Количество скрытых нейронных слоёв
	numTotalLayer	:= numHiddenLayer + 2					// Количество всех слоёв скрытых, входного и выходного
	numNeuron		:= []int{2,5,4,2}						// Количество нейронов для каждого слоя
	layer			:= make([]neuralLayer, numTotalLayer)	// Создаём срез нейронных слоёв

	// Создаем срезы для структуры нейронных слоёв
	for i := 0; i < numTotalLayer; i++ {
		layer[i].Neuron = make([]float64, numNeuron[i])
		layer[i].Error  = make([]float64, numNeuron[i])
		layer[i].Weight = make([][]float64, numNeuron[i])
		for j := 0; j < numNeuron[i] && i < numTotalLayer - 1; j++ {
			layer[i].Weight[j] = make([]float64, numNeuron[i + 1])
		}
	}

	// Входные параметры
	layer[0].Neuron = []float64{1.2,6.3}

	// Заполняем все веса случайными числами от 0.0 до 1.0
	fillWeight(layer[:])

	// Считаем значения нейрона в слое
	for i := 1; i < numTotalLayer; i++ {
		for j := 0; j < numNeuron[i]; j++ {
			layer[i].Neuron[j] = layer[i-1].calcNeuron()
		}
	}
	fmt.Println(layer)
}

// Функция заполняет все веса случайными числами от 0.0 до 1.0
func fillWeight(layer []neuralLayer) {
	for i := 0; i < len(layer) - 1; i++ {
		n := len(layer[i + 1].Neuron)
		for j := range layer[i].Neuron {
			for k := 0; k < n; k++ {
				layer[i].Weight[j][k] = rand.Float64()
			}
		}
	}
}

// Функция считает значения нейрона в слое
func (layer neuralLayer) calcNeuron() float64 {
	sum := 0.0
	for i := range layer.Neuron {
		for j := range layer.Weight[i] {
			sum += layer.Neuron[i] * layer.Weight[i][j]
		}
	}
	return getNeuralActivation(sum,0)
}

// Функция активации нейрона
func getNeuralActivation(sum float64, mode uint8) float64 {
	switch mode {
	default: fallthrough
	case 0: return 1 / (1 + math.Pow(math.E, -sum)) // Sigmoid
	case 1: // Leaky ReLu
		switch {
		case sum < 0: return 0.01 * sum
		case sum > 1: return 1 + 0.01 * (sum - 1)
		default: return sum
		}
	}
}