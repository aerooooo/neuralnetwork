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

	i, j := 0, 0
	var v string
	var array [][]float32
	array = make([][]float32, 1)
	for {
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
				array = make([][]float32, i + 1)
				for j, v = range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 32); err == nil {
						//array[i] = make([]float32, j + 1)
						//array[i][j] = float32(f)
						array[i] = append(array[i], float32(f))
					} else {
						log.Fatalln(err)
					}
				}
				i++
			} else {
				break
			}
		}
	}
	fmt.Println(i,j + 1, array[0][2])


	//mx.Training(input, data)

	err = mx.WriteWeight(datafile + ".weight")
	if err != nil {
		log.Fatal(err)
	}
}
