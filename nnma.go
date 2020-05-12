package main

import (
	"bufio"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

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

	//mx := new(nn.Matrix)
	var mx nn.Matrix
	mx.Mode   = nn.TANH
	mx.Rate   = .3
	mx.Bias   = 1
	mx.Limit  = .00001
	mx.Hidden = []int{11, 7}

	// Считываем данные из файла
	filename := "nnma/nnma_EURUSD_M60_2-5-8_0_0.dat"

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
						row = append(row, f / 1000) // 1000 - Коэфициент масштабирования данных, приводящих к промежутку от -1 до 1
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
	//fmt.Println(dataset[0],len(dataset),cap(dataset),dataset[len(dataset) - 1][0],dataset[len(dataset) - 1][1],dataset[len(dataset) - 1][2])

	// Обучение
	var (
		input	[]float64
		target	[]float64
		loss	float64
	)
	numInputBar  := 5
	numOutputBar := 3
	count  := 1
	//iter := 0
	//num  := 0
	//sum  := 0.

	for epoch := 0; epoch < 3; epoch++ {
		for i := numInputBar; i <=/*numInputBar*/len(dataset) - numOutputBar; i++ {
			input = getInputArray(dataset[i - numInputBar:i])
			//fmt.Println(input)
			target = getDataArray(dataset[i:i + numOutputBar])
			//fmt.Println(target)

			/*if !mx.IsInit {
				mx.IsInit = mx.Initializing(input, target)
			} else {
				copy(mx.Layer[0].Neuron, input)
				//fmt.Println(mx.Layer[0].Neuron)
			}
			for j := 0; j < 1; j++ {
				mx.CalcNeuron()
				loss = mx.CalcOutputError(target)
				mx.CalcError()
				mx.UpdateWeight()
				count = j
				//fmt.Println(loss)
			}*/


			count, loss = mx.Training(input, target)
			/*num += count
			sum += loss
			iter++*/
			/*count, loss = mx.Training(getMirror(input, target))
			num += count
			sum += loss
			iter++*/
		}
	}

	// Записываем данные вессов в файл
	err = mx.WriteWeight(filename + ".weight")
	if err != nil {
		log.Fatal(err)
	}

	// Вывод значений нейросети
	//mx.Print(num / iter, sum / float32(iter))
	mx.Print(count, loss)
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
func getDataArray(dataset [][]float64) []float64 {
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