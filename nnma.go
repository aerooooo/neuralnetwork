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
	numInputBar  := 5
	numOutputBar := 2
	dataScale    := 1000.  // Коэфициент масштабирования данных, приводящих к промежутку от -1 до 1
	start        := time.Now()

	//mx := new(nn.Matrix)
	var mx nn.Matrix
	mx.Mode   = nn.TANH
	mx.Rate   = .3
	mx.Bias   = 1
	mx.Limit  = .1 / dataScale // .1 / dataScale = .0001
	mx.Hidden = []int{20, 20, 20, 20, 20}

	// Считываем данные из файла
	filename := "c:/Users/teratron/AppData/Roaming/MetaQuotes/Terminal/0B5C5552DA53B624A3CF5DCF17492076/MQL4/Files/NNMA/nnma_EURUSD_M60_1-3-5_0_0.dat"

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
	maxEpoch := 10000
	for epoch := 1; epoch <= maxEpoch; epoch++ {
		startEpoch := time.Now()
		//for i := numInputBar; i <= len(dataset) - numOutputBar; i++ {
		for i := len(dataset) - numOutputBar - 100; i <= len(dataset) - numOutputBar; i++ {
			input  = getInputArray(dataset[i - numInputBar:i])
			target = getTargetArray(dataset[i:i + numOutputBar])

			// Если только знак
			//input  = getSignArray(input)
			//target = getSignArray(target)

			loss, count = mx.Training(input, target)

			// Mirror
			//loss, _ = mx.Training(getMirror(input, target))
		}
		endEpoch := time.Now()
		if epoch == 1 || epoch == maxEpoch {
			fmt.Printf("Epoch: %v,\tCount: %v, Elapsed time: %v\n", epoch, count, endEpoch.Sub(startEpoch))
		}

		// Test
		sum := 0.
		j := 0
		for i := len(dataset) - numOutputBar - 100; i <= len(dataset) - numOutputBar; i++ {
			   _ = mx.Querying(getInputArray(dataset[i - numInputBar:i]))
			sum += mx.CalcOutputError(getTargetArray(dataset[i:i + numOutputBar]), nn.MSE)
			j++
		}
		sum /= float64(j)
		startEpoch = time.Now()
		if epoch == 1 || epoch == maxEpoch || sum <= mx.Limit {
			fmt.Printf("Epoch: %v,\tCount: %v, Elapsed time: %v, Error: %.8f\n", epoch, count, startEpoch.Sub(endEpoch), sum)
			if sum <= mx.Limit {
				break
			}
		}
	}

	// Альтернативный принцип обучения
	//n := numInputBar
	/*n := len(dataset) - numOutputBar - 100
	for epoch := 1; epoch <= 100; epoch++ {
		for i := n; i <= len(dataset)-numOutputBar; i++ {
			for j := n; j < i; j++ {
				input = getInputArray(dataset[j-numInputBar : j])
				target = getDataArray(dataset[j : j+numOutputBar])
				loss, count = mx.Training(input, target)

				// Mirror
				loss, _ = mx.Training(getMirror(input, target))
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
	defer fmt.Printf("Elapsed time: %v\n", end.Sub(start))

	/*fmt.Println(nn.GetActivation(1.13, nn.SIGMOID))
	fmt.Println(nn.GetDerivative(nn.GetActivation(1.13, nn.SIGMOID), nn.SIGMOID) * -(-0.25))

	fmt.Println(nn.GetActivation(-0.53, nn.SIGMOID))
	fmt.Println(nn.GetDerivative(nn.GetActivation(-0.53, nn.SIGMOID), nn.SIGMOID) * -0.22 * 0.045)*/
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

// Зеркалит данные
func getMirror(input, target []float64) ([]float64, []float64) {
	for i := range input {
		input[i] *= -1
	}
	for i := range target {
		target[i] *= -1
	}
	return input, target
}

// Возвращает знак
func getSignArray(dataset []float64) []float64 {
	for i, v := range dataset {
		switch {
		case v < 0:
			dataset[i] = -1
		case v > 0:
			dataset[i] = 1
		default:
			dataset[i] = 0
		}
	}
	return dataset
}