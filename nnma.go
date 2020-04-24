package main

import "github.com/teratron/neuralnetwork/nn"

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


}
