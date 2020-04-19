/*
Большиство встроенных операций ввода-вывода не используют буфер. Это может иметь отрицательный эффект для производительности приложения. Для буферизации потоков чтения и записи в Go опредеелены ряд возможностей, которые сосредоточены в пакете bufio.

Запись через буфер
Для записи в источник данных через буфер в пакете bufio определен тип Writer. Чтобы записать данные, можно воспользоваться одним из его методов:

func (b *Writer) Write(p []byte) (nn int, err error)
func (b *Writer) WriteByte(c byte) error
func (b *Writer) WriteRune(r rune) (size int, err error)
func (b *Writer) WriteString(s string) (int, error)

Write(): записывает срез байтов
WriteByte(): записывает один байт
WriteRune(): записывает один объект типа rune
WriteString(): записывает строку

При выполнении этих методов данные вначале накапливаются в буфере, а чтобы сбросить их в источник данных, необходимо вызвать метод Flush().
Для создания потока вывода через буфер применяется функция bufio.NewWriter():

func NewWriter(w io.Writer) *Writer

Она принимает объект io.Writer - это может быть любой объект, в который идет запись: os.Stdout, файл и т.д. В качестве результата возвращается объект bufio.Writer.
В данном случае в файл через буферизированный поток вывода записываются две строки.
*/
package test

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	rows := []string{
		"Hello Go!",
		"Welcome to Golang",
	}

	file, err := os.Create("some.dat")
	writer := bufio.NewWriter(file)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer file.Close()

	for _, row := range rows {
		_, _ = writer.WriteString(row)    // запись строки
		_, _ = writer.WriteString("\n")   // перевод строки
	}
	_ = writer.Flush()       // сбрасываем данные из буфера в файл
}