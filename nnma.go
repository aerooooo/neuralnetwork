package main

import (
	"bufio"
	"fmt"
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
	mx.Limit  = .01
	mx.Hidden = []int{11, 7}
	//isMirror     := true
	numInputBar  := 3
	//numOutputBar := 3

	// Считываем данные из файла
	filename := "nnma/nnma_EURUSD_M60_2-5-8_0_0.dat"

	file, err := os.Open(filename)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	reader  := bufio.NewReader(file)
	dataset := make([][]float32, 0)

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
				var row []float32
				dataset = append(dataset, row)
				for _, v := range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 32); err == nil {
						row = append(row, float32(f))
					} else {
						log.Fatalln(err)
					}
				}
				dataset[i] = make([]float32, len(row))
				copy(dataset[i], row)
				i++
			} else {
				break
			}
		}
	}
	//fmt.Println(dataset[0],len(dataset),cap(dataset))
	//fmt.Println(dataset[len(dataset) - 1][0],dataset[len(dataset) - 1][1],dataset[len(dataset) - 1][2])
	//fmt.Println(io.SeekEnd)

	// Обучение
	var input  []float32
	//var output []float32
	//for epoch := 0; epoch < 4; epoch++ {
		for i := numInputBar; i <= numInputBar /*+ 1len(dataset)*/; i++ {
			d := getArray(dataset[i - numInputBar:i])
			/*d := make([]float32, 0)
			for _, r := range dataset[i - numInputBar:i] {
				d = append(d, r...)
			}*/
			//fmt.Println(d)

			k := len(d)
			input = make([]float32, k)
			for _, v := range d {
				k--
				input[k] = v
			}
			fmt.Println(input)



			d = getArray(dataset[i - numInputBar:i])
			k = len(d)
			fmt.Println(d)
		}
	//}

	/*for i, v := range dataset[numInputBar - 1:numInputBar][:] {
		fmt.Println(i, v)
	}*/

	// Вывод значений нейросети
	//mx.Print(0, 0)

	// Initialization



	//
	//mx.FillWeight()

	// Обучение
	/*for i := range dataset {
		for j, v := range dataset[i] {
			fmt.Println(j, v)
		}
	}*/
	/*
	for epoch := 0; epoch < 4; epoch++ {
	count, loss := mx.Training(input, data)
	}*/

	// Записываем данные вессов в файл
	/*err = mx.WriteWeight(filename + ".weight")
	if err != nil {
		log.Fatal(err)
	}*/
}

func getArray(dataset [][]float32) []float32 {
	d := make([]float32, 0)
	for _, r := range dataset {
		d = append(d, r...)
	}
	return d
}