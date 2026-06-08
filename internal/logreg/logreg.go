package logreg

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// Model es un clasificador de Regresión Logística binaria.
//
// API estilo Python/sklearn:
//
//	m := logreg.New(
//	    logreg.WithLearningRate(0.1),
//	    logreg.WithEpochs(200),
//	    logreg.WithWorkers(8),
//	)
//	m.Fit(xTrain, yTrain)
//	pred  := m.Predict(xTest)       // []float64 con 0.0 / 1.0
//	proba := m.PredictProba(xTest)  // []float64 con probabilidades
//	acc   := m.Score(xTest, yTest)
type Model struct {
	// --- Hiperparámetros de entrenamiento ---
	lr        float64 // tasa de aprendizaje (learning rate)
	epochs    int     // número de épocas
	l2        float64 // fuerza de regularización L2 (lambda)
	batchSize int     // tamaño de mini-batch; <=0 => batch completo
	nWorkers  int     // número de goroutines para el gradiente
	tol       float64 // tolerancia de convergencia (early stop); <=0 desactiva
	threshold float64 // umbral de decisión para Predict (por defecto 0.5)
	seed      int64   // semilla para el barajado de mini-batches
	verbose   bool    // imprime la pérdida durante el entrenamiento

	// --- Parámetros aprendidos ---
	weights []float64 // un peso por feature
	bias    float64   // término independiente (intercept)
	nFeat   int
	fitted  bool

	// LossHistory guarda la pérdida (binary cross-entropy) por época.
	LossHistory []float64
}

// Option configura el modelo al construirlo (patrón "functional options",
// el equivalente idiomático en Go a los argumentos con nombre de Python).
type Option func(*Model)

// WithLearningRate fija la tasa de aprendizaje.
func WithLearningRate(lr float64) Option { return func(m *Model) { m.lr = lr } }

// WithEpochs fija el número de épocas.
func WithEpochs(n int) Option { return func(m *Model) { m.epochs = n } }

// WithL2 fija la fuerza de regularización L2 (lambda).
func WithL2(l float64) Option { return func(m *Model) { m.l2 = l } }

// WithBatchSize fija el tamaño de mini-batch. <=0 => gradiente con batch completo.
func WithBatchSize(b int) Option { return func(m *Model) { m.batchSize = b } }

// WithWorkers fija cuántas goroutines colaboran en el cálculo del gradiente.
func WithWorkers(n int) Option {
	return func(m *Model) {
		if n < 1 {
			n = 1
		}
		m.nWorkers = n
	}
}

// WithTolerance fija el umbral de convergencia. Si la pérdida mejora menos
// que tol entre dos épocas, el entrenamiento se detiene. <=0 lo desactiva.
func WithTolerance(t float64) Option { return func(m *Model) { m.tol = t } }

// WithThreshold fija el umbral de decisión para Predict (por defecto 0.5).
func WithThreshold(t float64) Option { return func(m *Model) { m.threshold = t } }

// WithSeed fija la semilla del barajado de mini-batches (reproducibilidad).
func WithSeed(s int64) Option { return func(m *Model) { m.seed = s } }

// WithVerbose activa/desactiva el log de entrenamiento.
func WithVerbose(v bool) Option { return func(m *Model) { m.verbose = v } }

// New crea un modelo con valores por defecto razonables y aplica las opciones.
func New(opts ...Option) *Model {
	m := &Model{
		lr:        0.01,
		epochs:    100,
		l2:        0.0,
		batchSize: 0, // batch completo
		nWorkers:  runtime.NumCPU(),
		tol:       0.0,
		threshold: 0.5,
		seed:      42,
	}
	for _, opt := range opts {
		opt(m)
	}
	if m.nWorkers < 1 {
		m.nWorkers = 1
	}
	return m
}

// Fit entrena el modelo con X (n muestras x d features) e y (n etiquetas 0/1).
func (m *Model) Fit(X [][]float64, y []float64) error {
	if len(X) == 0 {
		return fmt.Errorf("logreg: X está vacío")
	}
	if len(X) != len(y) {
		return fmt.Errorf("logreg: X tiene %d filas pero y tiene %d", len(X), len(y))
	}

	n := len(X)
	m.nFeat = len(X[0])
	for i, row := range X {
		if len(row) != m.nFeat {
			return fmt.Errorf("logreg: la fila %d tiene %d features, se esperaban %d", i, len(row), m.nFeat)
		}
	}

	// Inicializamos los parámetros en cero.
	m.weights = make([]float64, m.nFeat)
	m.bias = 0
	m.LossHistory = m.LossHistory[:0]

	// Índices de todas las muestras (se barajan si usamos mini-batches).
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	rng := rand.New(rand.NewSource(m.seed))

	prevLoss := math.Inf(1)
	for epoch := 0; epoch < m.epochs; epoch++ {
		var epochLoss float64

		if m.batchSize <= 0 || m.batchSize >= n {
			// --- Batch completo: una sola actualización por época ---
			gradW, gradB, loss := m.gradient(X, y, indices)
			m.update(gradW, gradB)
			epochLoss = loss
		} else {
			// --- Mini-batch: barajamos y recorremos en bloques ---
			rng.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
			var batches int
			for start := 0; start < n; start += m.batchSize {
				end := start + m.batchSize
				if end > n {
					end = n
				}
				gradW, gradB, loss := m.gradient(X, y, indices[start:end])
				m.update(gradW, gradB)
				epochLoss += loss
				batches++
			}
			epochLoss /= float64(batches)
		}

		m.LossHistory = append(m.LossHistory, epochLoss)

		if m.verbose && (epoch%max(1, m.epochs/10) == 0 || epoch == m.epochs-1) {
			fmt.Printf("época %4d/%d  loss=%.6f\n", epoch+1, m.epochs, epochLoss)
		}

		// Parada temprana por convergencia.
		if m.tol > 0 && math.Abs(prevLoss-epochLoss) < m.tol {
			if m.verbose {
				fmt.Printf("convergió en la época %d (Δloss < %g)\n", epoch+1, m.tol)
			}
			break
		}
		prevLoss = epochLoss
	}

	m.fitted = true
	return nil
}

// partial guarda el resultado parcial calculado por un worker.
// Cada worker tiene su PROPIO partial => no hay escritura compartida.
type partial struct {
	gradW []float64 // gradiente parcial respecto a los pesos
	gradB float64   // gradiente parcial respecto al bias
	loss  float64   // suma de la pérdida en el chunk
}

// gradient calcula el gradiente promedio (con regularización L2) sobre las
// muestras indicadas por `indices`, repartiendo el trabajo entre m.nWorkers
// goroutines. Aquí vive el paralelismo.
func (m *Model) gradient(X [][]float64, y []float64, indices []int) (gradW []float64, gradB, loss float64) {
	n := len(indices)
	workers := m.nWorkers
	if workers > n {
		workers = n // no tiene sentido tener más workers que muestras
	}

	// Cada worker escribe en results[id]: posiciones distintas => sin races.
	results := make([]partial, workers)
	chunk := (n + workers - 1) / workers // división con redondeo hacia arriba

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(id, s, e int) {
			defer wg.Done()

			// Buffers LOCALES del worker (no compartidos).
			lg := make([]float64, m.nFeat)
			var lb, ll float64

			for k := s; k < e; k++ {
				i := indices[k]
				xi := X[i]

				// Solo LECTURA de los pesos compartidos (seguro: nadie escribe ahora).
				p := sigmoid(m.dot(xi) + m.bias)
				errTerm := p - y[i]

				for j, xj := range xi {
					lg[j] += errTerm * xj
				}
				lb += errTerm
				ll += crossEntropy(y[i], p)
			}
			results[id] = partial{gradW: lg, gradB: lb, loss: ll}
		}(w, start, end)
	}

	// Barrera de sincronización: esperamos TODOS los gradientes parciales.
	wg.Wait()

	// --- Reducción: el hilo principal suma los resultados parciales ---
	gradW = make([]float64, m.nFeat)
	for _, r := range results {
		if r.gradW == nil { // chunk vacío
			continue
		}
		for j := range gradW {
			gradW[j] += r.gradW[j]
		}
		gradB += r.gradB
		loss += r.loss
	}

	// Promedio sobre las n muestras del batch.
	invN := 1.0 / float64(n)
	for j := range gradW {
		gradW[j] = gradW[j]*invN + m.l2*invN*m.weights[j] // + término de regularización L2
	}
	gradB *= invN
	loss = loss*invN + m.l2Penalty(invN)
	return gradW, gradB, loss
}

// update aplica el paso de gradiente descendente (único escritor de los pesos).
func (m *Model) update(gradW []float64, gradB float64) {
	for j := range m.weights {
		m.weights[j] -= m.lr * gradW[j]
	}
	m.bias -= m.lr * gradB
}

// PredictProba devuelve la probabilidad P(y=1|x) para cada fila de X.
func (m *Model) PredictProba(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		out[i] = sigmoid(m.dot(x) + m.bias)
	}
	return out
}

// Predict devuelve la clase predicha (0.0 o 1.0) según el umbral configurado.
func (m *Model) Predict(X [][]float64) []float64 {
	proba := m.PredictProba(X)
	out := make([]float64, len(proba))
	for i, p := range proba {
		if p >= m.threshold {
			out[i] = 1
		}
	}
	return out
}

// Score devuelve la exactitud (accuracy) del modelo sobre X, y.
func (m *Model) Score(X [][]float64, y []float64) float64 {
	pred := m.Predict(X)
	correct := 0
	for i := range y {
		if pred[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}

// Weights devuelve una copia de los pesos aprendidos.
func (m *Model) Weights() []float64 {
	w := make([]float64, len(m.weights))
	copy(w, m.weights)
	return w
}

// Bias devuelve el término independiente aprendido.
func (m *Model) Bias() float64 { return m.bias }

// dot calcula el producto punto pesos·x.
func (m *Model) dot(x []float64) float64 {
	var s float64
	for j, w := range m.weights {
		s += w * x[j]
	}
	return s
}

// l2Penalty es el término de regularización añadido a la pérdida.
func (m *Model) l2Penalty(invN float64) float64 {
	if m.l2 == 0 {
		return 0
	}
	var s float64
	for _, w := range m.weights {
		s += w * w
	}
	return 0.5 * m.l2 * invN * s
}

// sigmoid es la función logística, en versión numéricamente estable.
func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	ez := math.Exp(z)
	return ez / (1.0 + ez)
}

// crossEntropy es la pérdida de una muestra (binary cross-entropy), con un
// epsilon para evitar log(0).
func crossEntropy(y, p float64) float64 {
	const eps = 1e-12
	p = math.Min(math.Max(p, eps), 1-eps)
	return -(y*math.Log(p) + (1-y)*math.Log(1-p))
}
