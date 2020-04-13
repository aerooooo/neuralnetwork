package main

import (
	"fmt"

	"github.com/fogleman/gg"
)

/*type Matrix struct {
	Size	int
	Layer	[]Layer
}*/

type Layer struct {
	Size	int
	Node	[]Node
}

type Node struct {
	Size	int
	Neuron	float32
	Error	float32
	Weight	[]float32
}

func main() {
	layer := Layer{}
	//layer.Node
	//matrix := Matrix{}
	/*layer  := make([]Layer, 5)
	node  :=
	//node   := Node
	layer[0].Size = 5
	layer[0].Node*/
	var node Node

	fmt.Println(layer.Node," ",node)

	dc := gg.NewContext(1000, 1000)
	dc.DrawCircle(200, 200, 100)
	dc.SetRGB(255, 50, 50)
	dc.Fill()
	_ = dc.SavePNG("out.png")
}
