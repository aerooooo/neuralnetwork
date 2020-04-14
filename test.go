package main

import "fmt"

type NNMatrix struct {
	Size	int			// Количество слоёв в нейросети (Input + Hidden + Output)
	Data	[]float32	// Обучающий набор с которым будет сравниваться выходной слой
	Input	[]float32	// Входные параметры
	Layer	[]NNLayer	// Коллекция полей слоя
	Weight	[]NNWeight	// Коллекция массива весов
}

type NNLayer struct {
	Size	int			// Количество нейронов в слое
	Node	[]NNNode	// Коллекция полей нейрона
}

type NNNode struct {
	Neuron	float32		// Значения нейрона
	Error	float32		// Значение ошибки
}

type NNWeight struct {
	Weight	[][]float32
}

func main() {
	matrix := NNMatrix{
		Size:	4,
		Data:	[]float32{6.3, 3.2},
		Input:	[]float32{1.2, 6.3},
		Layer:	[]NNLayer{
			{ // 1 - Входной слой
				Size: 2,
				Node: []NNNode{
					{Error: nil},
					{Error: nil},
				},
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
		{Size: 4, Node: []NNNode{}},
		{2, []NNNode{}},
	}

	fmt.Println(layer[0].Node[0].Neuron, matrix)
	//layer.Node
	//matrix := Matrix{}
	/*layer  := make([]Layer, 5)
	node  :=
	//node   := Node
	layer[0].Size = 5
	layer[0].Node*/
	//var node Neuron

	//fmt.Println(layer[0].Node," ",node)
}
