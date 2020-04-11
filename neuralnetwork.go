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
	//bias			:= 1												// Смещение
	numLayer := 4                                              // Количество всех слоёв: входного, выходного и скрытых
	indLayer := numLayer - 1                                   // Индекс последнего слоя нейросети
	dataSet := []float32{6.3, 1.2}                             // Обучающий набор с которым будет сравниваться выходной слой
	layer := make([]NeuralLayer, numLayer)                     // Создаём срез нейронных слоёв
	layer[0].Node = []float32{1.2, 6.3}                        // Входные параметры
	numNeuron := []int{len(layer[0].Node), 5, 4, len(dataSet)} // Количество нейронов для каждого слоя

	// Создаем срезы для структуры нейронных слоёв
	for i := 0; i < numLayer; i++ {
		if i > 0 {
			layer[i].Node = make([]float32, numNeuron[i])
			layer[i].Error = make([]float32, numNeuron[i])
		}
		layer[i].Weight = make([][]float32, numNeuron[i])
		for j := 0; j < numNeuron[i] && i < indLayer; j++ {
			layer[i].Weight[j] = make([]float32, numNeuron[i+1])
		}
	}

	// Заполняем все веса случайными числами от 0.0 до 1.0
	fillWeight(layer)

	// Считаем значения нейрона в слое
	for i := 1; i < numLayer; i++ {
		for j := 0; j < numNeuron[i]; j++ {
			layer[i].Node[j] = layer[i-1].calcNeuron()
		}
		fmt.Println(i, " ", layer[i].Node)
	}

	// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
	for i := 0; i < numNeuron[indLayer]; i++ {
		layer[indLayer].Error[i] = dataSet[i] - layer[indLayer].Node[i]
	}
	fmt.Println("Error:\n", indLayer, " ", layer[indLayer].Error)

	// Вычисляем ошибки нейронов в скрытых слоях
	for i := indLayer - 1; i > 0; i-- {
		for j := 0; j < numNeuron[i]; j++ {
			layer[i].Error[j] = layer[i].calcError(layer[i+1].Error)
		}
		fmt.Println(i, " ", layer[i].Error)
	}

	fmt.Println(layer)
}

// Функция заполняет все веса случайными числами от 0.0 до 1.0
func fillWeight(layer []NeuralLayer) {
	for i := 0; i < len(layer)-1; i++ {
		n := len(layer[i+1].Node)
		for j := range layer[i].Node {
			for k := 0; k < n; k++ {
				layer[i].Weight[j][k] = rand.Float32()
			}
		}
	}
}

// Функция вычисления значения нейрона в слое
func (layer NeuralLayer) calcNeuron() float32 {
	var sum float32 = 0
	for i := range layer.Node {
		for j := range layer.Weight[i] {
			sum += layer.Node[i] * layer.Weight[i][j]
		}
	}
	return getNeuralActivation(sum, 0)
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (layer NeuralLayer) calcError(error []float32) float32 {
	var sum float32 = 0
	//fmt.Println(len(layer.Weight),"   ",len(error))
	for i := range layer.Weight {
		for j := range layer.Weight[i] {
			sum += error[j] * layer.Weight[i][j]
		}
	}
	return sum
}

// Функция активации нейрона
func getNeuralActivation(sum float32, mode uint8) float32 {
	switch mode {
	default:
		fallthrough
	case 0:
		return (float32)(1 / (1 + math.Pow(math.E, (float64)(-sum)))) // Sigmoid
	case 1: // Leaky ReLu
		switch {
		case sum < 0:
			return 0.01 * sum
		case sum > 1:
			return 1 + 0.01*(sum-1)
		default:
			return sum
		}
	}
}
