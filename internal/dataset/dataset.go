package dataset

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
)

// TrainTestSplit divide X, y en conjuntos de entrenamiento y prueba.
//
// testRatio es la fracción destinada a test (p. ej. 0.2 = 20%).
// seed controla el barajado para que la división sea reproducible.
//
// Devuelve: xTrain, xTest, yTrain, yTest (mismo orden que en sklearn).
func TrainTestSplit(X [][]float64, y []float64, testRatio float64, seed int64) (xTrain, xTest [][]float64, yTrain, yTest []float64) {
	n := len(X)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	rand.New(rand.NewSource(seed)).Shuffle(n, func(i, j int) {
		idx[i], idx[j] = idx[j], idx[i]
	})

	nTest := int(math.Round(float64(n) * testRatio))
	for k, i := range idx {
		if k < nTest {
			xTest = append(xTest, X[i])
			yTest = append(yTest, y[i])
		} else {
			xTrain = append(xTrain, X[i])
			yTrain = append(yTrain, y[i])
		}
	}
	return xTrain, xTest, yTrain, yTest
}

// StandardScaler estandariza cada feature a media 0 y desviación 1.
// Imprescindible para que el gradiente descendente converja bien.
//
// Uso (estilo sklearn):
//
//	sc := dataset.NewStandardScaler()
//	xTrain = sc.FitTransform(xTrain)
//	xTest  = sc.Transform(xTest)   // se reusan media/std del train
type StandardScaler struct {
	Mean []float64
	Std  []float64
}

// NewStandardScaler crea un escalador sin ajustar.
func NewStandardScaler() *StandardScaler { return &StandardScaler{} }

// Fit calcula media y desviación estándar de cada feature a partir de X.
func (s *StandardScaler) Fit(X [][]float64) *StandardScaler {
	if len(X) == 0 {
		return s
	}
	d := len(X[0])
	s.Mean = make([]float64, d)
	s.Std = make([]float64, d)

	for _, row := range X {
		for j, v := range row {
			s.Mean[j] += v
		}
	}
	n := float64(len(X))
	for j := range s.Mean {
		s.Mean[j] /= n
	}
	for _, row := range X {
		for j, v := range row {
			diff := v - s.Mean[j]
			s.Std[j] += diff * diff
		}
	}
	for j := range s.Std {
		s.Std[j] = math.Sqrt(s.Std[j] / n)
		if s.Std[j] == 0 {
			s.Std[j] = 1 // evita dividir por cero en features constantes
		}
	}
	return s
}

// Transform aplica la estandarización aprendida y devuelve una matriz nueva.
func (s *StandardScaler) Transform(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, row := range X {
		nr := make([]float64, len(row))
		for j, v := range row {
			nr[j] = (v - s.Mean[j]) / s.Std[j]
		}
		out[i] = nr
	}
	return out
}

// FitTransform es Fit seguido de Transform.
func (s *StandardScaler) FitTransform(X [][]float64) [][]float64 {
	return s.Fit(X).Transform(X)
}

// Save guarda la media y la desviación aprendidas en un archivo JSON. Hace
// falta para estandarizar nuevas entradas igual que en el entrenamiento al
// predecir sin reentrenar.
func (s *StandardScaler) Save(path string) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadStandardScaler reconstruye un escalador ya ajustado desde un archivo
// guardado con Save.
func LoadStandardScaler(path string) (*StandardScaler, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	s := &StandardScaler{}
	if err := json.Unmarshal(data, s); err != nil {
		return nil, err
	}
	return s, nil
}
