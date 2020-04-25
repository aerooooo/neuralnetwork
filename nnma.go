package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
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

	//
	datafile := "ma_EURUSD_#1768.dat"
	file, err := os.Open(datafile)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	i := 0
	for {
		if line, err := reader.ReadString('\n'); err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		} else {
			line = strings.Trim(line,"\n")
			if len(line) > 0 {
				i++
				fmt.Println(line)
			} else {
				break
			}
		}
	}
	fmt.Println(i,reader)
	var array [][]float32
	array = make([][]float32, i, i)
	array[0] = make([]float32, i, i)

	//reader = bufio.NewReader(file)

	reader.Reset(file)
	line, err := reader.ReadString('\n')
	//fmt.Println(line)
	/*if err != nil {
		log.Println(err)
	} else {*/
		for i = range strings.Split(line, "\t") {
			i++
		}
	//}

	//fmt.Println(i,reader)


	// Считываем данные из файла
	/*file, err := os.Open(datafile)
	if err != nil {
		log.Fatalln(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	for i := 0;; {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		} else {
			line = strings.Trim(line,"\n")
			if len(line) > 0 {
				for j, v := range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 32); err == nil {
						Weight[i][j] = float32(f)
					} else {
						log.Fatalln(err)
					}
				}
				i++
			} else {
				break
			}
		}
	}*/
}
