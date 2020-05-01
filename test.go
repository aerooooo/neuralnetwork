package main

import (
	"fmt"
)

const (
	DEFRATE	float32	= .3
	MINLOSS	float32	= .001
)

type Checker interface {
	Checking() float32
}

type MX struct {
	Rate
	Bias
	Limit
}

type (
	Rate  float32
	Bias  float32
	Limit float32
)

//var MX = (*MX)(nil)

func (b Bias) Checking() float32 {
	switch {
	case b < 0: return 0
	case b > 1: return 1
	default: 	return float32(b)
	}
}

func (l Limit) Checking() float32 {
	switch {
	case l < 0: return MINLOSS
	default:	return float32(l)
	}
}

func (r Rate) Checking() float32 {
	switch {
	case r < 0 || r > 1: return DEFRATE
	default:			 return float32(r)
	}
}

func Check(c ...Checker) {
	for _, v := range c {
		fmt.Println(v,v.Checking())
	}
}

func main() {
	m := MX{}

	//fmt.Println(m.Bias)
	m.Bias  = Bias(2.1)
	m.Limit = Limit(-0.1)
	m.Rate  = Rate(8.1)

	//D := [...]Checker{B, L, R}
	Check(m.Bias, m.Limit, m.Rate)

	fmt.Println(m.Bias.Checking())
	fmt.Println(m.Limit.Checking())
	fmt.Println(m.Rate.Checking())

	//fmt.Println(bbb)
}