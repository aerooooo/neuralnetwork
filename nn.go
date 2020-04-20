package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// Коллекция параметров матрицы
type NNMatrix struct {
	Size	int				// Количество слоёв в нейросети (Input + Hidden + Output)
	Mode	uint8			// Индетефикатор функции активации: 0 - Sigmoid, 1 - Leaky ReLu, 2 - Tanh
	Bias	float32			// Нейрон смещения
	Ratio 	float32			// Коэффициент обучения, от 0 до 1
	Data	[]float32		// Обучающий набор с которым будет сравниваться выходной слой
	Layer	[]NNLayer		// Коллекция слоя
	Link	[]NNWeight		// Коллекция весов
}

// Коллекция параметров нейронного слоя
type NNLayer struct {
	Size	int				// Количество нейронов в слое
	Neuron	[]float32		// Значения нейрона
	Error	[]float32		// Значение ошибки
}

// Коллекция параметров весов
type NNWeight struct {
	Size	[]int			// Количество связей весов {X, Y}, X - входной (предыдущий) слой, Y - выходной (следующий) слой
	Weight	[][]float32		// Значения весов
}

func main() {
	var (
		collision	float32
		bias		float32	= 0
		ratio		float32	= .5
		input	= []float32{1.2, 6.3}	// Входные параметры
		data	= []float32{6.3, 3.2}	// Обучающий набор с которым будет сравниваться выходной слой
		hidden	= []int{5, 4}			// Массив количеств нейронов в каждом скрытом слое
		mode		uint8 = 0			// Индетефикатор функции активации
	)

	// Инициализация нейросети
	matrix := new(NNMatrix)
	matrix.InitNN(mode, bias, ratio, input, data, hidden)

	// Заполняем все веса случайными числами от -0.5 до 0.5
	matrix.FillWeight()

	// Обучение нейронной сети за какое-то количество эпох
	for i := 0; i < 100; i++ {
		matrix.CalcNeuron()                  // Вычисляем значения нейронов в слое
		collision = matrix.CalcOutputError() // Вычисляем ошибки между обучающим набором и полученными выходными нейронами
		matrix.CalcError()                   // Вычисляем ошибки нейронов в скрытых слоях
		matrix.UpdWeight()                   // Обновление весов
	}

	err := matrix.WriteWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	err = matrix.ReadWeight("weight.dat")
	if err != nil {
		log.Fatal(err)
	}

	// Вывод значений нейросети
	matrix.PrintNN(collision)
}

//
func Get() {

}

//
func GetOutput(bias float32, input []float32, matrix *NNMatrix) (output []float32) {
	//matrix.CalcNeuron()                  // Вычисляем значения нейронов в слое
	return output
}

// Функция инициализации матрицы
func (m *NNMatrix) InitNN(mode uint8, bias float32, ratio float32, input []float32, data []float32, hidden []int) {
	var i, j, index int
	layer := []int{len(input)}
	for _, v := range hidden {
		layer = append(layer, v)
	}
	layer = append(layer, len(data))
	m.Size  = len(layer)
	index   = m.Size - 1
	m.Layer = make([]NNLayer,  m.Size)
	m.Link  = make([]NNWeight, index)
	m.Data  = make([]float32,  index)
	m.Ratio = ratio
	m.Mode  = mode
	for i, j = range layer {
		m.Layer[i].Size = j
	}
	switch {
	case bias < 0:	m.Bias = 0
	case bias > 1:	m.Bias = 1
	default: 		m.Bias = bias
	}
	for i = 0; i < m.Size; i++ {
		// Создаем срезы для структуры нейронных слоёв и весов
		m.Layer[i].Neuron = make([]float32, m.Layer[i].Size)
		if i > 0 {
			m.Layer[i].Error = make([]float32, m.Layer[i].Size)
		}
		if i < index {
			m.Layer[i].Neuron = append(m.Layer[i].Neuron, m.Bias)
			m.Link[i].Size    = []int{m.Layer[i].Size + 1, m.Layer[i + 1].Size}
			m.Link[i].Weight  = make([][]float32, m.Link[i].Size[0])
			for j = 0; j < m.Link[i].Size[0]; j++ {
				m.Link[i].Weight[j] = make([]float32, m.Link[i].Size[1])
			}
		}
	}
	copy(m.Layer[0].Neuron, input)
	copy(m.Data, data)
}

// Функция заполняет все веса случайными числами от -0.5 до 0.5
func (m *NNMatrix) FillWeight() {
	for i := 0; i < m.Size - 1; i++ {
		n := m.Link[i].Size[0] - 1
		for j := 0; j < m.Link[i].Size[0]; j++ {
			for k := 0; k < m.Link[i].Size[1]; k++ {
				if j == n && m.Bias == 0 {
					m.Link[i].Weight[j][k] = 0
				} else {
					m.Link[i].Weight[j][k] = rand.Float32() - .5
				}
			}
		}
	}
}

// Функция вычисления значения нейронов в слое
func (m *NNMatrix) CalcNeuron() {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j := 0; j < m.Layer[i].Size; j++ {
			var sum float32 = 0
			for k, v := range m.Layer[n].Neuron {
				sum += v * m.Link[n].Weight[k][j]
			}
			m.Layer[i].Neuron[j] = GetActivation(sum, m.Mode)
		}
	}
}

// Функция вычисления ошибки выходного нейрона
func (m *NNMatrix) CalcOutputError() (collision float32) {
	collision = 0
	j := m.Size - 1
	for i, v := range m.Layer[j].Neuron {
		m.Layer[j].Error[i] = (m.Data[i] - v) * GetDerivative(v, 0)
		collision += float32(math.Pow(float64(m.Layer[j].Error[i]), 2))
	}
	return collision
}

// Функция вычисления ошибки нейронов в скрытых слоях
func (m *NNMatrix) CalcError() {
	for i := m.Size - 2; i > 0; i-- {
		for j := 0; j < m.Layer[i].Size; j++ {
			var sum float32 = 0
			for k, v := range m.Layer[i + 1].Error {
				sum += v * m.Link[i].Weight[j][k]
			}
			m.Layer[i].Error[j] = sum * GetDerivative(m.Layer[i].Neuron[j], 0)
		}
	}
}

// Функция обновления весов
func (m *NNMatrix) UpdWeight() {
	for i := 1; i < m.Size; i++ {
		n := i - 1
		for j, v := range m.Layer[i].Error {
			for k, p := range m.Layer[n].Neuron {
				m.Link[n].Weight[k][j] += m.Ratio * v * p * GetDerivative(m.Layer[i].Neuron[j], 0)
			}
		}
	}
}

// Функция активации нейрона
func GetActivation(value float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return float32(1 / (1 + math.Pow(math.E, float64(-value)))) // Sigmoid
	case 1: // Leaky ReLu
		switch {
		case value < 0: return 0.01 * value
		case value > 1: return 1 + 0.01 * (value - 1)
		default:	  	return value
		}
	case 2: return float32(2 / (1 + math.Pow(math.E, float64(-2 * value))) - 1) // Tanh - гиперболический тангенс
	}
}

// Функция производной активации
func GetDerivative(value float32, mode uint8) float32 {
	switch mode {
	default: fallthrough
	case 0: return value * (1 - value)
	case 1: return 1
	}
}

// Функция вывода результатов нейросети
func (m *NNMatrix) PrintNN(collision float32) {
	t := "Layer"
	n := m.Size - 1
	for i := 0; i < m.Size; i++ {
		if i == len(m.Layer) - 1 {
			t = " Output layer"
		}
		fmt.Println(i, t, "size: ", m.Layer[i].Size)
		fmt.Println("Neurons:\t", m.Layer[i].Neuron)
		fmt.Println("Errors:\t\t", m.Layer[i].Error)
	}
	fmt.Println("Weights:")
	for i := 0; i < n; i++ {
		fmt.Println(m.Link[i].Weight)
	}
	fmt.Println("Total Error:\t", collision)
}

// Записываем данные вессов в файла
func (m *NNMatrix) WriteWeight(filename string) error {
	file, err := os.Create(filename)
	writer := bufio.NewWriter(file)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer file.Close()

	for i := 0; i < m.Size - 1; i++ {
		for j := 0; j < m.Link[i].Size[0]; j++ {
			for k := 0; k < m.Link[i].Size[1]; k++ {
				_, err = writer.WriteString(strconv.FormatFloat(float64(m.Link[i].Weight[j][k]), 'f', -1, 32)) // Запись строки
				if k < m.Link[i].Size[1] - 1 {
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
func (m *NNMatrix) ReadWeight(filename string) error {
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
			if len(line) > 0 {
				for k, v := range strings.Split(line, "\t") {
					if f, err := strconv.ParseFloat(v, 32); err == nil {
						//fmt.Printf("%v, %v, %v, %T, %v\n", i, j, k, f, float32(f))
						//fmt.Println(i, j, k, float32(f))
						m.Link[i].Weight[j][k] = float32(f)
					} else {
						log.Fatal(err)
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