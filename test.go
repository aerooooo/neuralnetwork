package main

import (
	"fmt"
)

type phoneReader string

func (ph *phoneReader) Read(p *[]byte) /*(int, error)*/ {
	h := *ph
	z := *p
	count := 0
	for i := 0; i < len(h); i++ {
		if h[i] >= '0' && h[i] <= '9' {
			z[count] = h[i]
			//fmt.Println(count, p[count])
			count++

		}
	}
	//return count, io.EOF
}

func main() {
	phone1 := phoneReader("+1(234)567 9010")
	phone2 := phoneReader("+2-345-678-12-35")

	buffer := make([]byte, len(phone1))
	/*count, err := */phone1.Read(&buffer)
	fmt.Println(/*count, err, */phone1, buffer, string(buffer))     // 12345679010

	buffer = make([]byte, len(phone2))
	/*_, _ = */phone2.Read(&buffer)
	fmt.Println(phone2, string(buffer))     // 23456781235
}