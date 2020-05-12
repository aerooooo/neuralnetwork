package main

import (
	"log"

	"github.com/teratron/neuralnetwork/nn"
)

func main() {
	var (
		count int
		loss  float64
		input  = []float64{1.2, 6.3} // Входные параметры
		target = []float64{6.3, 3.2} // Обучающий набор с которым будет сравниваться выходной слой
		hidden = []int{5, 4}         // Массив количеств нейронов в каждом скрытом слое
	)

	// Инициализация нейросети
	var matrix nn.Matrix
	matrix.InitMatrix(nn.SIGMOID, 1, .5, .01, input, target, hidden)

	// Обучение нейронной сети за какое-то количество эпох
	for epoch := 0; epoch < 4; epoch++ {
		count, loss = matrix.Training(input, target)
	}

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
