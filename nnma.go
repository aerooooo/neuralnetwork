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

//type array [][]float32

func main() {
	/*mx := nn.Matrix{
		Mode:	nn.TANH,
		Rate:	.3,
		Bias:	1,
		Limit:	.01,
		Hidden:	[]int{5, 4},
	}*/

	mx := new(nn.Matrix)
	mx.Mode   = nn.TANH
	mx.Rate   = .3
	mx.Bias   = 1
	mx.Limit  = .01
	mx.Hidden = []int{5, 4}

	//matrix.Init(mode, rate, bias, limit, input, data, hidden)

	// Считываем данные из файла
	datafile := "ma_EURUSD_#1768.dat"

	file, err := os.Open(datafile)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	var dataset [][]float32
	//dataset := array{}

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

	//fmt.Println(len(dataset[0]))
	//fmt.Println(len(dataset),cap(dataset))
	//fmt.Println(dataset[len(dataset) - 1][0])

	// Обучение
	for i := range dataset {
		for j, v := range dataset[i] {
			fmt.Println(j,v)
		}
	}

	//count, loss := mx.Training(input, data)

	// Записываем данные вессов в файл
	/*err = mx.WriteWeight(datafile + ".weight")
	if err != nil {
		log.Fatal(err)
	}*/
}
