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
	/*mx := nn.Matrix{
		Mode:	nn.TANH,
		Rate:	.3,
		Bias:	1,
		Epoch:	4,
		Limit:	.01,
		Hidden:	[]int{5, 4},
	}*/
	var (
		input	[]float64
		target	[]float64
		loss	float64
		count	int
	)
	numInputBar  := 8	// 5
	numOutputBar := 3
	dataScale    := 1000.  // Коэфициент масштабирования данных, приводящих к промежутку от -1 до 1
	start        := time.Now()

	//mx := new(nn.Matrix)
	var mx nn.Matrix
	mx.Mode   = nn.TANH
	mx.Rate   = .3
	mx.Bias   = 1
	mx.Limit  = 1 / dataScale // .1 / dataScale = .0001
	mx.Hidden = []int{60, 60, 60, 60, 60}

	// Считываем данные из файла
	filename := "c:/Users/teratron/AppData/Roaming/MetaQuotes/Terminal/0B5C5552DA53B624A3CF5DCF17492076/MQL4/Files/NNMA/nnma_EURUSD_M60_1-3-5_1-3-5.dat"
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	reader  := bufio.NewReader(file)
	dataset := make([][]float64, 0)

	for i := 0;; {
		if line, err := reader.ReadString('\n'); err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		} else {
			line = strings.Trim(line,"\n")
			if strings.HasSuffix(line, "\r") {
				line = strings.Trim(line,"\r")
			}
			if len(line) > 0 {
				var row []float64
				dataset = append(dataset, row)
				for _, v := range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 64); err == nil {
						row = append(row, f / dataScale)
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
	//weight   := new([][]float64)
	maxEpoch := 100000
	minError := 1.
	for epoch := 1; epoch <= maxEpoch; epoch++ {
		startEpoch := time.Now()
		//for i := numInputBar; i <= len(dataset) - numOutputBar; i++ {
		for i := len(dataset) - numOutputBar - 5000; i <= len(dataset) - numOutputBar; i++ {
			input  = getInputArray(dataset[i - numInputBar:i])
			target = getTargetArray(dataset[i:i + numOutputBar])
			loss, count = mx.Training(input, target)
		}

		// Testing
		sum := 0.
		j := 0
		//for i := numInputBar; i <= len(dataset) - numOutputBar; i++ {
		for i := len(dataset) - numOutputBar - 5000; i <= len(dataset) - numOutputBar; i++ {
			   _ = mx.Querying(getInputArray(dataset[i - numInputBar:i]))
			loss = mx.CalcOutputError(getTargetArray(dataset[i:i + numOutputBar]), nn.MSE)
			sum += loss
			j++
			/*if loss > mx.Limit {
				break
			}*/
		}

		// Средняя ошибка за всю эпоху
		sum /= float64(j)

		//
		/*if loss > mx.Limit {
			if epoch == 1 || epoch == 10 || epoch % 1000 == 0 || epoch == maxEpoch {
				fmt.Printf("+++++++++ Epoch: %v\tError: %.8f\n", epoch, sum)
			}
			continue
		}*/

		// Минимальная средняя ошибка
		if sum < minError && epoch >= 1000 {
			minError = sum
			fmt.Println("--------- Epoch:", epoch, "\tmin avg error:", minError)
			/*if epoch >= 10000 {
				mx.CopyWeight(weight)
			}*/
		}

		//
		if epoch == 1 || epoch == 10 || epoch % 1000 == 0 || epoch == maxEpoch || sum <= mx.Limit {
			fmt.Printf("Epoch: %v\tCount: %v\tElapsed time: %v / %v\tError: %.8f\n", epoch, count, time.Now().Sub(startEpoch), time.Now().Sub(start), sum)
			if sum <= mx.Limit {
				break
			}
		}
	}

	// Альтернативный принцип обучения
	/*maxEpoch := 10000
	//n := numInputBar
	n := len(dataset) - numOutputBar - 100
	for epoch := 1; epoch <= maxEpoch; epoch++ {
		startEpoch := time.Now()
		for i := n; i <= len(dataset)-numOutputBar; i++ {
			for j := n; j < i; j++ {
				input  = getInputArray(dataset[j-numInputBar : j])
				target = getTargetArray(dataset[j : j+numOutputBar])
				loss, count = mx.Training(input, target)

				// Mirror
				//loss, count = mx.Training(getMirror(input, target))
			}
		}

		// Testing
		sum := 0.
		j := 0
		for i := n; i <= len(dataset) - numOutputBar; i++ {
			   _ = mx.Querying(getInputArray(dataset[i - numInputBar:i]))
			sum += mx.CalcOutputError(getTargetArray(dataset[i:i + numOutputBar]), nn.MSE)
			j++
		}
		sum /= float64(j)
		endEpoch := time.Now()
		if epoch == 1 || epoch == maxEpoch || sum <= mx.Limit {
			fmt.Printf("Epoch: %v\tCount: %v\tElapsed time: %v\tError: %.8f\n", epoch, count, endEpoch.Sub(startEpoch), sum)
			if sum <= mx.Limit {
				break
			}
		}
	}*/

	// Записываем данные вессов в файл
	err = mx.WriteWeight(filename + ".weight")
	if err != nil {
		log.Fatal(err)
	}

	// Вывод значений нейросети
	mx.Print(count, loss)

	// Elapsed time
	end := time.Now()
	fmt.Printf("Elapsed time: %v\n", end.Sub(start))
}

// Возвращает массив входных параметров
func getInputArray(dataset [][]float64) []float64 {
	d := make([]float64, 0)
	for i := len(dataset) - 1; i >= 0; i-- {
		d = append(d, dataset[i]...)
	}
	return d
}

// Возвращает массив эталонных данных
func getTargetArray(dataset [][]float64) []float64 {
	d := make([]float64, 0)
	for _, r := range dataset {
		d = append(d, r[0])
	}
	return d
}