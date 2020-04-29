package test

import (
	"fmt"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	_, _ = fmt.Fprint(w, "Hello, world!")
}

func main() {
	http.HandleFunc("/", handler)
	_ = http.ListenAndServe("8080", nil)
}