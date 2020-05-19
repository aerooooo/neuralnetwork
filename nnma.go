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
	numInputBar  := 7
	numOutputBar := 2
	dataScale    := 1000.  // Коэфициент масштабирования данных, приводящих к промежутку от -1 до 1
	start        := time.Now()

	//mx := new(nn.Matrix)
	var mx nn.Matrix
	mx.Mode   = nn.TANH
	mx.Rate   = .3
	mx.Bias   = 1
	mx.Limit  = .000001
	mx.Hidden = []int{20, 20, 10}

	// Считываем данные из файла
	filename := "nnma/nnma_EURUSD_M60_1-5_0_0.dat"

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
	//fmt.Println(len(dataset), cap(dataset), dataset[0], dataset[len(dataset) - 1][0], dataset[len(dataset) - 1][1]/*,dataset[len(dataset) - 1][2]*/)

	// Обучение
	/*iter := 0
	num  := 0
	sum  := 0.*/

	for epoch := 0; epoch < 100; epoch++ {
		for i := /*numInputBar*/len(dataset) - numOutputBar - 10; i <= len(dataset) - numOutputBar; i += 1 {
			input = getInputArray(dataset[i - numInputBar:i])
			//fmt.Println(input)
			target = getDataArray(dataset[i:i + numOutputBar])
			//fmt.Println(target)

			// Если только знак
			//input = getSignArray(input)
			//fmt.Println(input)
			//target = getSignArray(target)
			//fmt.Println(target)

			count = 1
			for count < 1000/*nn.MAXITER*/ {
				loss, _ = mx.Training(input, target)
				//if loss <= mx.Limit || loss <= nn.MINLOSS {
				if loss <= 0 || loss <= nn.MINLOSS {
					break
				}
				/*num += count
				sum += loss
				iter++*/
				count++
				/*count, loss = mx.Training(getMirror(input, target))
				num += count
				sum += loss
				iter++*/
			}
		}
	}

	// Записываем данные вессов в файл
	err = mx.WriteWeight(filename + ".weight")
	if err != nil {
		log.Fatal(err)
	}

	// Вывод значений нейросети
	//mx.Print(num / iter, sum / float64(iter))
	mx.Print(count, loss)

	// Elapsed time
	t := time.Now()
	elapsed := t.Sub(start)
	defer fmt.Printf("Elapsed time: %v\n", elapsed)



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

//
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