package main

import "fyne.io/fyne/app"
import "fyne.io/fyne/widget"

func main() {
	a := app.New()

	w := a.NewWindow("Hello")
	w.SetContent(widget.NewVBox(
		widget.NewLabel("Hello Fyne!"),
		widget.NewButton("Quit", func() {
			a.Quit()
		})))

	w.ShowAndRun()
}