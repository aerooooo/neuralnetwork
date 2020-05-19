package main

import (
	"fmt"
	"log"
	"time"

	"github.com/teratron/neuralnetwork/nn"
)

func main() {
	var (
		count int
		loss  float64
		input  = []float64{-.2, .63}  // Входные параметры
		target = []float64{.63, -.32} // Обучающий набор с которым будет сравниваться выходной слой
		hidden = []int{5, 4}          // Массив количеств нейронов в каждом скрытом слое
		start  = time.Now()
	)

	// Инициализация нейросети
	var matrix nn.Matrix
	matrix.InitMatrix(nn.TANH, 1, .5, .01, input, target, hidden)

	// Обучение нейронной сети за какое-то количество эпох
	for epoch := 0; epoch < 1; epoch++ {
		loss, _ = matrix.Training(input, target)
	}
	//matrix.FillWeight()

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

	// Elapsed time
	t := time.Now()
	elapsed := t.Sub(start)
	defer fmt.Printf("Elapsed time: %v\n", elapsed)

	//fmt.Printf("%v\n", nn.MAXITER)
}
