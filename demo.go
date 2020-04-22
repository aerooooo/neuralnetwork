package main

import (
	"log"

	"github.com/teratron/neuralnetwork/nn"
)

func init() {
}

func main() {
	var (
		loss	float32
		bias	float32 = 0
		ratio	float32 = .5
		input	= []float32{1.2, 6.3}	// Входные параметры
		data	= []float32{6.3, 3.2}	// Обучающий набор с которым будет сравниваться выходной слой
		hidden	= []int{5, 4}			// Массив количеств нейронов в каждом скрытом слое
		mode	= nn.SIGMOID				// Идентификатор функции активации
	)

	// Инициализация нейросети
	var matrix nn.Matrix
	matrix.Init(mode, bias, ratio, input, data, hidden)

	// Заполняем все веса случайными числами от -0.5 до 0.5
	matrix.FillWeight()

	// Обучение нейронной сети за какое-то количество эпох
	for i := 0; i < 100; i++ {
		matrix.CalcNeuron()					// Вычисляем значения нейронов в слое
		loss = matrix.CalcOutputError()		// Вычисляем ошибки между обучающим набором и полученными выходными нейронами
		matrix.CalcError()					// Вычисляем ошибки нейронов в скрытых слоях
		matrix.UpdWeight()					// Обновление весов
	}

	err := matrix.WriteWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	err = matrix.ReadWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	//nn.Get()
	//nn.GetOutput(bias, input, &matrix)

	// Вывод значений нейросети
	matrix.Print(loss)
}