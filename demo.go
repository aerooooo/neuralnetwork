package main

import (
	"log"

	"github.com/teratron/neuralnetwork/nn"
)

func main() {
	var (
		input	= []float32{1.2, 6.3}	// Входные параметры
		data	= []float32{6.3, 3.2}	// Обучающий набор с которым будет сравниваться выходной слой
		hidden	= []int{5, 4}			// Массив количеств нейронов в каждом скрытом слое
	)

	// Инициализация нейросети
	var matrix nn.Matrix
	matrix.InitMatrix(nn.SIGMOID, 1, .5, .01, input, data, hidden...)

	// Обучение нейронной сети за какое-то количество эпох
	count, loss := matrix.Training(input, data)

	err := matrix.WriteWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	err = matrix.ReadWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	// Вывод значений нейросети
	matrix.Print(count, loss)
}