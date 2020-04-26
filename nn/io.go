package nn

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// Функция вывода результатов нейросети
func (m *Matrix) Print(loss float32) {
	var i int
	t := "Layer"
	for i = 0; i < m.Size; i++ {
		if i == m.Index {
			t = "Output layer"
		}
		fmt.Printf("%v %s size: %v\n", i, t, m.Layer[i].Size)
		fmt.Println("Neurons:\t", m.Layer[i].Neuron)
		fmt.Println("Errors:\t\t", m.Layer[i].Error)
	}
	fmt.Println("Weights:")
	for i = 0; i < m.Index; i++ {
		fmt.Println(m.Synapse[i].Weight)
	}
	fmt.Println("Total Error:\t", loss)
}

// Записываем данные вессов в файл
func (m *Matrix) WriteWeight(filename string) error {
	file, err := os.Create(filename)
	writer := bufio.NewWriter(file)
	if err != nil {
		os.Exit(1)
	}
	defer file.Close()

	for i := 0; i < m.Index; i++ {
		for j := 0; j < m.Synapse[i].Size[0]; j++ {
			for k := 0; k < m.Synapse[i].Size[1]; k++ {
				_, err = writer.WriteString(strconv.FormatFloat(float64(m.Synapse[i].Weight[j][k]), 'f', -1, 32)) // Запись строки
				if k < m.Synapse[i].Size[1] - 1 {
					_, err = writer.WriteString("\t") // Разделяем значения
				} else {
					_, err = writer.WriteString("\n") // Перевод строки
				}
			}
		}
		if i < m.Size - 2 {
			_, err = writer.WriteString("\n") // Перевод строки
		}
	}
	return writer.Flush()	// Сбрасываем данные из буфера в файл
}

// Считываем данные вессов из файла
func (m *Matrix) ReadWeight(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	for i, j := 0, 0;; {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return err
			}
		} else {
			line = strings.Trim(line,"\n")
			if strings.HasSuffix(line, "\r") {
				line = strings.Trim(line,"\r")
			}
			if len(line) > 0 {
				for k, v := range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 32); err == nil {
						m.Synapse[i].Weight[j][k] = float32(f)
					} else {
						return err
					}
				}
				j++
			} else {
				j = 0
				i++
			}
		}
	}
	return err
}
