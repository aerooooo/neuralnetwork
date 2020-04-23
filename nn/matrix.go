package nn

import "fmt"

// Training
// Action

type NN interface {
	//Init()
	FillWeight()
	CalcNeuron()
	/*CalcOutputError()
	CalcError()
	UpdWeight()
	Print()*/
}

func Measure(g NN) {

	fmt.Println("--start--")
	g.FillWeight()
	fmt.Println("--end--")
	g.CalcNeuron()
	//fmt.Println(g.Print(.002))
}

//var nn NN

/*func SetCurrentApp(current NN) {
	//nn = current
}*/
/*
// CurrentApp returns the current application, for which there is only 1 per process.
func CurrentApp() NN {
	return nn
}*/