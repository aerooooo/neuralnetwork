package main

import (
	"fmt"
	"fyne.io/fyne/app"
	"fyne.io/fyne/widget"
	"net/url"
)

func main() {
	app2 := app.New()

	w := app2.NewWindow("Hello")
	w.SetContent(widget.NewLabel("Hello Fyne!"))

	bugURL, _ := url.Parse("https://github.com/fyne-io/fyne/issues/new")
	w.SetContent(widget.NewHyperlink("Report a bug", bugURL))

	//w.ShowAndRun()
	// or
	w.Show()
	app2.Run()
	tidyUp()
}

func tidyUp() {
	fmt.Println("Exited")
}