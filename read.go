/*
Чтение через буфер
Для чтения из источника данных через буфер в пакете bufio определен тип Reader. Для чтения данных можно воспользоваться одним из его методов:

func (b *Reader) Read(p []byte) (n int, err error)
func (b *Reader) ReadByte() (byte, error)
func (b *Reader) ReadBytes(delim byte) ([]byte, error)
func (b *Reader) ReadLine() (line []byte, isPrefix bool, err error)
func (b *Reader) ReadRune() (r rune, size int, err error)
func (b *Reader) ReadSlice(delim byte) (line []byte, err error)
func (b *Reader) ReadString(delim byte) (string, error)

Read(p []byte): считывает срез байтов и возвращает количество прочитанных байтов
ReadByte(): считывает один байт
ReadBytes(delim byte): считывает срез байтов из потока, пока не встретится байт delim
ReadLine(): считывает строку в виде среза байт
ReadRune(): считывает один объект типа rune
ReadSlice(delim byte): считывает срез байтов из потока, пока не встретится байт delim
ReadString(delim byte): считывает строку, пока не встретится байт delim

Для создания потока ввода через буфер применяется функция bufio.NewReader():

func NewReader(rd io.Reader) *Reader

Она принимает объект io.Reader - это может быть любой объект, с которого производится чтение: os.Stdin, файл и т.д. В качестве результата возвращается объект bufio.Reader.
В данном случае идет считывания из ранее записанного файла. Для этого объект файла os.File передается в функцию bufio.NewReader, на основании которого создается объект bufio.Reader. Поскольку идет построчное считывание, то каждая строка считывается из потока, пока не будет обнаружен символ перевода строки \n.
*/
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

func main() {
	file, err := os.Open("some.dat")
	if err != nil {
		fmt.Println("Unable to open file:", err)
		return
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				fmt.Println(err)
				return
			}
		}
		fmt.Print(line)
	}
}