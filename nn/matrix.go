package nn

type NN interface {
}

type Checker interface {
	Checking() float32
}

// Collection of neural network matrix parameters
type Matrix struct {
	Init	bool		// Флаг выполнения инициализации матрицы
	Size	int			// Количество слоёв в нейросети (Input + Hidden + Output)
	Index	int			// Индекс выходного (последнего) слоя нейросети
	Mode	uint8		// Идентификатор функции активации
	Bias	float32		// Нейрон смещения: от 0 до 1
	Rate 	float32		// Коэффициент обучения, от 0 до 1
	Limit	float32		// Минимальный (достаточный) уровень средней квадратичной суммы ошибки при обучения
	Hidden	[]int		// Массив количеств нейронов в каждом скрытом слое
	Layer	[]Layer		// Коллекция слоя
	Synapse	[]Synapse	// Коллекция весов связей
}

// Collection of neural layer parameters
type Layer struct {
	Size	int			// Количество нейронов в слое
	Neuron	[]float32	// Значения нейрона
	Error	[]float32	// Значение ошибки
}

// Collection of weight parameters
type Synapse struct {
	Size	[]int		// Количество связей весов {X, Y}, X - входной (предыдущий) слой, Y - выходной (следующий) слой
	Weight	[][]float32	// Значения весов
}

type (
	FloatType	float32
	Bias		FloatType
	Rate		FloatType
	Limit		FloatType
)