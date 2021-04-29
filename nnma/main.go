package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/teratron/neuralnetwork/nn"
)

func main() {
	var (
		input  []float64
		target []float64
		loss   float64
		count  int
	)

	numInputBar := 20
	numOutputBar := 1
	dataScale := 1000. // Коэфициент масштабирования данных, приводящих к промежутку от -1 до 1
	start := time.Now()

	var mx = &nn.Matrix{}
	mx.ModeActivation = nn.TANH
	mx.Rate = .3
	mx.Bias = 1
	mx.Limit = .1 / dataScale // .1 / dataScale = .0001
	mx.Hidden = []int{20, 20}

	// Считываем данные из файла
	filename := "c:/Users/teratron/AppData/Roaming/MetaQuotes/Terminal/0B5C5552DA53B624A3CF5DCF17492076/MQL4/Files/NNMA/nnma_EURUSD_M60_5_0.dat"
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	//
	reader := bufio.NewReader(file)
	dataset := make([][]float64, 0)
	for i := 0; ; {
		if line, err := reader.ReadString('\n'); err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		} else {
			line = strings.Trim(line, "\n")
			if strings.HasSuffix(line, "\r") {
				line = strings.Trim(line, "\r")
			}
			if len(line) > 0 {
				var row []float64
				dataset = append(dataset, row)
				for _, v := range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 64); err == nil {
						row = append(row, f/dataScale)
					} else {
						log.Fatalln(err)
					}
				}
				dataset[i] = make([]float64, len(row))
				copy(dataset[i], row)
				i++
			} else {
				break
			}
		}
	}

	// Обучение
	var weight []nn.Synapse
	minError := 1.
	maxEpoch := 1000000
	limit := mx.Limit //* 10

	startDataset := len(dataset) - numOutputBar
	endDataset := startDataset - 10 //len(dataset)-numOutputBar-10
	startDataset -= 1000            //len(dataset) - numOutputBar - 1000

	for epoch := 1; epoch <= maxEpoch; epoch++ {
		startEpoch := time.Now()
		//for i := numInputBar; i <= len(dataset) - numOutputBar; i++ {
		for i := startDataset; i <= endDataset; i++ {
			input = getInputArray(dataset[i-numInputBar : i])
			target = getTargetArray(dataset[i : i+numOutputBar])
			loss, count = mx.Training(input, target)
		}

		// Testing
		sum, num := 0., 0.
		//for i := numInputBar; i <= len(dataset) - numOutputBar; i++ {
		for i := startDataset; i <= endDataset; i++ {
			_ = mx.Querying(getInputArray(dataset[i-numInputBar : i]))
			loss = mx.CalcOutputError(getTargetArray(dataset[i:i+numOutputBar]), nn.MSE)
			sum += loss
			num++
		}

		// Средняя ошибка за всю эпоху
		sum /= num

		// Веса нейросети копируются при минимальной средней ошибки
		if sum < minError && epoch >= 1000 {
			minError = sum
			fmt.Println("\t- Epoch:", epoch, "\tmin avg error:", minError)

			// Копируем в буффер веса с наименьшей ошибкой
			if epoch >= 10000 {
				weight = mx.Synapse
			}
		}

		// Выход из эпох обучения при достижении минимального уровня ошибки
		if epoch == 1 || epoch == 10 || epoch%100 == 0 || epoch == maxEpoch || sum <= mx.Limit {
			fmt.Printf("Epoch: %d\tCount: %d\tElapsed time: %v / %v\tError: %.8f\n", epoch, count, time.Now().Sub(startEpoch), time.Now().Sub(start), sum)
			if sum <= limit {
				loss = sum
				break
			}
		}
	}

	// Возвращаем из буффера веса
	if weight != nil {
		mx.Synapse = weight
	}

	// Записываем данные вессов в файл
	err = mx.WriteWeight(filename + ".weight")
	if err != nil {
		log.Fatal(err)
	}

	// Вывод значений нейросети
	mx.Print(count, loss)

	// Elapsed time
	fmt.Printf("Elapsed time: %v\n", time.Now().Sub(start))
}

// getInputData returns input data
// Возвращает массив входных параметров
func getInputArray(dataset [][]float64) []float64 {
	d := make([]float64, 0)
	for i := len(dataset) - 1; i >= 0; i-- {
		d = append(d, dataset[i]...)
	}
	return d
}

// getTargetData returns reference data
// Возвращает массив эталонных данных
func getTargetArray(dataset [][]float64) []float64 {
	d := make([]float64, 0)
	for _, r := range dataset {
		d = append(d, r[0])
	}
	return d
}
