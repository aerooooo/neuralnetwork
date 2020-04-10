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
	Weight	struct{
		weight []float64
	}
}

func main() {
	numHiddenLayer	:= 2									// Количество скрытых нейронных слоёв
	numTotalLayer	:= numHiddenLayer + 2					// Количество всех слоёв скрытых, входного и выходного
	numNeuron		:= []int{2,5,4,2}						// Количество нейронов для каждого слоя
	layer			:= make([]neuralLayer, numTotalLayer)	// Создаём срез нейронных слоёв
	layer.Weight	:= make([]neuralLayer, numNeuron[0])
	//
	for i := 0; i < numTotalLayer; i++ {
		layer[i].Neuron = make([]float64, numNeuron[i])
		layer[i].Error  = make([]float64, numNeuron[i])
		for j := 0; j < numNeuron[i]; j++ {
			fmt.Println(layer[i].Weight[j])
			//layer[i].Weight[j].weight = make([]float64, numNeuron[i])
		}
	}
	layer[0].Neuron  = []float64{1.2,6.3}					// Входные параметры
	fmt.Println(layer)
	// Заполняем все веса случайными числами от 0.0 до 1.0
	//fillWeight(layer[:])

	//
	/*for i := 1; i < numTotalLayer; i++ {
		n := 5//len(layer[i].Neuron)
		for j := 0; j < n; j++ {
			layer[i].Neuron[j] = layer[i-1].calcNeuron()
		}
	}
	fmt.Println(layer)*/
}

// Заполняем все веса случайными числами от 0.0 до 1.0
func fillWeight(layer []neuralLayer) {
	for i := 0; i < len(layer) - 1; i++ {
		n := len(layer[i + 1].Neuron)
		//fmt.Println(n," ",len(layer[i].Neuron))
		for j := range layer[i].Neuron {
			layer[i].Weight[j].weight = make([]float64, n)
			for k := 0; k < n; k++ {
				layer[i].Weight[j].weight[k] = rand.Float64()
				fmt.Println(i," ",j," ",k)
			}
		}
	}
}

// Функция считает значения нейрона в слое
func (layer neuralLayer) calcNeuron() float64 {
	sum := 0.0
	for i := range layer.Neuron {
		for j := range layer.Weight[i].weight {
			sum += layer.Neuron[i] * layer.Weight[i].weight[j]
		}
	}
	return getNeuralActivation(sum,0)
}

// Функция считает значения нейронов в слое
func (layer neuralLayer) getNeuralLayer() float64 {
	sum := 0.0
	for i := range layer.Neuron {
		for j := range layer.Weight[i].weight {
			sum += layer.Neuron[i] * layer.Weight[i].weight[j]
		}
		//layer[i].Input[0] =
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