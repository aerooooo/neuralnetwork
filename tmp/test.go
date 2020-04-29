package test

import (
	"fmt"
	"strconv"
)

// Конвертирует float32 в []byte
func ConvertFloat32ToByte(value float32) []byte {
	return []byte(strconv.FormatFloat(float64(value), 'f', -1, 32))
}

type NNMatrix struct {
	Size	int			// Количество слоёв в нейросети (Input + Hidden + Output)
	Data	[]float32	// Обучающий набор с которым будет сравниваться выходной слой
	Input	[]float32	// Входные параметры
	Layer	*[]NNLayer	// Коллекция полей слоя
	Weight	*[]NNWeight	// Коллекция массива весов
}

type NNLayer struct {
	Size	int			// Количество нейронов в слое
	Neuron	[]float32	// Значения нейрона
	Error	[]float32	// Значение ошибки
	Bias	float32		// Нейрон смещения
}

type NNWeight struct {
	Size	[2]int		// Количество связей весов {X, Y}, X - входной (предыдущий) слой, Y - выходной (следующий) слой
	Weight	[][]float32	// Значения весов
}

func main() {
	matrix := NNMatrix{
		Size:  4,
		Data:  []float32{6.3, 3.2},
		Input: []float32{1.2, 6.3},
	}

	layer := make([]NNLayer, matrix.Size)
	layer = []NNLayer{
		{Size: 2, Bias: 1},
		{Size: 5, Bias: 1},
		{Size: 4, Bias: 1},
		{Size: 2},
	}
	for i := 0; i < matrix.Size; i++ {
		layer[i].Neuron = make([]float32, layer[i].Size)
	}
	copy(layer[0].Neuron, matrix.Input)

	matrix.Layer =  &layer

	fmt.Println(matrix)
	fmt.Println(layer)
	fmt.Println(matrix.Layer)
	fmt.Println(layer[0].Neuron)

	/*matrix := NNMatrix{
		Size:	4,
		Data:	[]float32{6.3, 3.2},
		Input:	[]float32{1.2, 6.3},
		Layer:	[]NNLayer{
			{ // 1 - Входной слой
				Size: 2,
				Node: []NNNode{},
			},
			{ // 2
				Size: 5,
				Node: []NNNode{
					{0.1, 0.2},
					{0.2, 0.3},
					{0.3, 0.4},
					{0.4, 0.5},
					{0.5, 0.6},
				},
			},
			{Size: 4, Node: []NNNode{}},  // 3
			{Size: 2, Node: []NNNode{}},  // 4 - Выходной слой
		},
		Weight: []NNWeight{

		},
	}

	layer := make([]NNLayer, matrix.Size)
	layer = []NNLayer{
		{Size: 2, Bias: 1, Node: []NNNode{{0.4, 0.5}, {0.4, 0.5}}},
		{
			Size: 5,
			Node: []NNNode{
				{0.1, 0.2},
				{0.2, 0.3},
				{0.3, 0.4},
				{0.4, 0.5},
				{0.5, 0.6},
			},
		},
		{Size: 4, Bias: 1, Node: []NNNode{}},
		{Size: 2, Bias: 0, Node: []NNNode{}},
	}

	weight := make([]NNWeight, matrix.Size - 1)
	weight = []NNWeight{
		{
			Weight: [][]float32{
				{0.1, 0.2, 0.3, 0.4, 0.5},
				{0.1, 0.2, 0.3, 0.4, 0.5},
			},
		},
		{Weight: [][]float32{{0.1, 0.2, 0.3, 0.4, 0.5}, {0.1, 0.2, 0.3, 0.4, 0.5}}},
		{Weight: [][]float32{{0.1, 0.2, 0.3, 0.4, 0.5}, {0.1, 0.2, 0.3, 0.4, 0.5}}},
	}*/

	/*fmt.Println(layer[0].Neuron)
	fmt.Println(matrix)
	fmt.Println(weight[1].Weight[0][0])*/

	//matrix.Layer[0].Node[0].Neuron = 0.1

	//matrix1 := matrix.layer

	//fmt.Println(matrix{Layer: layer[0]}) //.Neuron= copy(matrix.Layer[0].Node[0].Neuron, matrix.Input)

}
