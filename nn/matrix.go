package nn

type Lister interface {
	Enumerate()
}

type Array struct {}

func (a Array) Enumerate() {
}

type Checker interface {
	Checking() float32
}

type (
	Rate  float32
	Bias  float32
	Limit float32
)

type NN interface {
	//Init()
	//FillWeight()
	//CalcNeuron()
	/*CalcOutputError()
	CalcError()
	UpdWeight()
	Print()*/
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