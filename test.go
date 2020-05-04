package main

import "fmt"

type Lister interface {
	Enumerate()
}

type Array struct {
	Bias
}

type Bias float64

func (a Array) Enumerate() {
	fmt.Println(a.Bias)
}
func (b Bias) Enumerate() {
	fmt.Println(b)
}

func main() {
	//a := Array{}
	var b Lister

	c := Array{ /*Bias(.5)*/ }
	b = Bias(.1)

	c.Enumerate()
	b.Enumerate()

	//fmt.Println(a.Bias)

	//a.Bias = Bias(0.1)

	//fmt.Println(a.Bias)
}

/*type NN interface {
	Init()
	FillWeight()
	CalcNeuron()
	CalcOutputError()
	CalcError()
	UpdWeight()
	Print()
}*/
//var nn NN
/*func SetCurrentApp(current NN) {
	//nn = current
}*/
/*
// CurrentApp returns the current application, for which there is only 1 per process.
func CurrentApp() NN {
	return nn
}*/
