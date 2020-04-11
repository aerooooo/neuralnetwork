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
	var i, j, k int
	//biasNeuron	:= 1								// Нейрон смещения
	//ratioLearn	:= .5								// Коэффициент обучения
	inputSet	:= []float32{1.2, 6.3}				// Входные параметры
	dataSet		:= []float32{6.3, 1.2}				// Обучающий набор с которым будет сравниваться выходной слой
	numNeuron	:= []int{5, 4, len(dataSet)}		// Количество нейронов для каждого слоя, исключая входной слой
	numInput	:= len(inputSet)					// Количество входных нейронов
	numLayer	:= len(numNeuron)					// Количество скрытых слоёв и выходного слоя
	indLayer	:= numLayer - 1						// Индекс последнего слоя нейросети
	layer 		:= make([]NeuralLayer, numLayer)	// Создаём срез нейронных слоёв

	// Первая проходка по нейронным слоям
	for i = 0; i < numLayer; i++ {
		// Создаем срезы для структуры нейронных слоёв
		layer[i].Node   = make([]float32,   numNeuron[i])
		layer[i].Error  = make([]float32,   numNeuron[i])
		layer[i].Weight = make([][]float32, numNeuron[i])
		for j = 0; j < numNeuron[i]; j++ {
			if i > 0 {
				k = numNeuron[i - 1]
			} else {
				k = numInput
			}
			layer[i].Weight[j] = make([]float32, k)
		}

		// Заполняем все веса случайными числами от -0.5 до 0.5
		layer[i].fillWeight()
	}

	// Считаем значения нейронов в слое
	layer[0].calcNeuron(&inputSet)
	for i = 1; i < numLayer; i++ {
		layer[i].calcNeuron(&layer[i - 1].Node)
	}

	// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
	for i = 0; i < numNeuron[indLayer]; i++ {
		layer[indLayer].Error[i] = (dataSet[i] - layer[indLayer].Node[i]) * layer[indLayer].Node[i] * (1 - layer[indLayer].Node[i])
	}
	//e = (float32)(math.Pow((float64)(), 2))

	// Вычисляем ошибки нейронов в скрытых слоях
	for i = indLayer; i > 0; i-- {
		j = i - 1
		layer[j].Error = layer[i].calcError(&layer[j].Node)
	}

	// Обновление весов
	//$W[$x,$y] = $W[$x,$y] + $k * $No[$y,1] * $Ni[$x,0] * $No[$y,0] * (1-$No[$y,0])
	//$W[$x,$y] = $W[$x,$y] + $k * $No[$y,1] * $Ni[$x,0]

	//Weight = Weight + ratioLearn * y * (1 - y) * inputSet

	// Вывод значений нейросети
	printLayer(&layer)
}

// Функция заполняет все веса случайными числами от -0.5 до 0.5
func (layer *NeuralLayer) fillWeight() {
	l := *layer
	for i := range l.Weight {
		for j := range l.Weight[i] {
			l.Weight[i][j] = rand.Float32() - .5
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
func (layer *NeuralLayer) calcError(node *[]float32) []float32 {
	l := *layer
	n := *node
	k := len(n)
	e := make([]float32, k)
	for i := 0; i < k; i++ {
		var sum float32 = 0
		for j := range l.Error {
			sum += l.Error[j] * l.Weight[j][i]
		}
		e[i] = sum * n[i] * (1 - n[i])
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
func printLayer(layer *[]NeuralLayer) {
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