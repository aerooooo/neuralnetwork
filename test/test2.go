package test

import "fmt"

type Vehicle interface {
	move()
}

type Car struct { model string}
type Aircraft struct { model string}

func (c Car) move() {
	fmt.Printf("%T %v едет\n", c.model, c.model)
}

func (a Aircraft) move() {
	fmt.Printf("%T %v летит\n", a.model, a.model)
}

func main() {
	var tesla Vehicle = Car{"Tesla"}
	volvo  := Car{"Volvo"}
	boeing := Aircraft{"Boeing"}

	fmt.Printf("%T %v едет\n", tesla, tesla)

	vehicles := [...]Vehicle{tesla, volvo, boeing}
	for _, vehicle := range vehicles {
		vehicle.move()
	}
}