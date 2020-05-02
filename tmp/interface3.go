package tmp

import (
	"fmt"
)

const (
	DEFRATE	float32	= .3
	MINLOSS	float32	= .001
)

type Checker2 interface {
	Checking2() float32
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

func (b *Bias) Checking2() float32 {
	switch {
	case *b < 0: return 0
	case *b > 1: return 1
	default: 	return float32(*b)
	}
}

func (r Rate) Checking2() float32 {
	switch {
	case r < 0 || r > 1: return DEFRATE
	default:			 return float32(r)
	}
}

func (l Limit) Checking2() float32 {
	switch {
	case l < 0: return MINLOSS
	default:	return float32(l)
	}
}

func Check(c ...Checker2) {
	for _, v := range c {
		fmt.Println(v, v.Checking2())
	}
}

func main() {
	m := MX{}

	m.Bias  = Bias(2.1)
	m.Limit = Limit(-0.1)
	m.Rate  = Rate(8.1)

	//D := [...]Checker2{B, L, R}
	Check(&m.Bias, m.Limit, m.Rate)

	var ch Checker2
	fmt.Println(ch)

	//fmt.Println(m.Bias.Checking2())
	//fmt.Println(m.Limit.Checking2())
	//fmt.Println(m.Rate.Checking2())
}