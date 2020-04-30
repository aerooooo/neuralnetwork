package main

import (
	"log"

	"github.com/teratron/neuralnetwork/nn"
)

func main() {
	var (
		loss	float32
		input	= []float32{1.2, 6.3}	// Входные параметры
		data	= []float32{6.3, 3.2}	// Обучающий набор с которым будет сравниваться выходной слой
		hidden	= []int{5, 4}			// Массив количеств нейронов в каждом скрытом слое
	)

	// Инициализация нейросети
	var matrix nn.Matrix
	err := matrix.InitMatrix(nn.SIGMOID, .5, 1, .001, input, data, hidden)
	if err != nil {
		log.Fatal(err)
	}

	// Заполняем все веса случайными числами от -0.5 до 0.5
	matrix.FillWeight()

	// Обучение нейронной сети за какое-то количество эпох
	for i := 0; i < nn.MAXITER; i++ {
		matrix.CalcNeuron()					// Вычисляем значения нейронов в слое
		// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
		if loss = matrix.CalcOutputError(data); loss <= matrix.Limit || loss <= nn.MINLOSS {
			break
		}
		matrix.CalcError()					// Вычисляем ошибки нейронов в скрытых слоях
		matrix.UpdWeight()					// Обновление весов
	}

	err = matrix.WriteWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	err = matrix.ReadWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	//nn.GetOutput(bias, input, &matrix)

	// Вывод значений нейросети
	matrix.Print(loss)

	//mm := new(nn.Matrix)
	//nn.Measure(mm)
}