package nn

// Training
// Action

type NN interface {
	//
	FillWeight()
	CalcNeuron()
}

//var nn NN

/*func SetCurrentApp(current NN) {
	nn = current
}

// CurrentApp returns the current application, for which there is only 1 per process.
func CurrentApp() NN {
	return nn
}*/