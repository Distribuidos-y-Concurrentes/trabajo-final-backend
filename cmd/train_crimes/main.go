package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/dataset"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/logreg"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/metrics"
)

// featureCols son las columnas de X, en orden. Cada nombre debe existir como
// cabecera en el CSV. Mantener este slice es la única fuente de verdad de qué
// entra al modelo (cambiar aquí cambia todo el pipeline).
var featureCols = []string{
	"district",
	"grid_lat",
	"grid_lon",
	"hour",
	"weekday",
	"month",
	"is_night",
	"location_encoded",
	"domestic", // opcional; quítalo de la lista si no quieres usarlo
}

// targetCol es la variable a predecir (Y).
const targetCol = "is_high_risk"

func main() {
	csvPath := flag.String("csv", "cmd/cleaning/crimes_clean.csv", "ruta al CSV limpio")
	epochs := flag.Int("epochs", 300, "número de épocas")
	lr := flag.Float64("lr", 0.5, "tasa de aprendizaje")
	l2 := flag.Float64("l2", 0.001, "regularización L2 (lambda)")
	testRatio := flag.Float64("test", 0.2, "fracción para test")
	threshold := flag.Float64("threshold", 0.5, "umbral de decisión para la clase positiva")
	seed := flag.Int64("seed", 42, "semilla (reproducibilidad)")
	workers := flag.Int("workers", runtime.NumCPU(), "goroutines para el gradiente")
	modelPath := flag.String("model", "crime_model.json", "ruta donde guardar/cargar los pesos del modelo")
	scalerPath := flag.String("scaler", "crime_scaler.json", "ruta donde guardar/cargar el StandardScaler")
	loadOnly := flag.Bool("load", false, "cargar modelo+scaler guardados y evaluar SIN reentrenar")
	flag.Parse()

	fmt.Printf("CPUs disponibles: %d\n", runtime.NumCPU())
	fmt.Printf("Cargando %q ...\n", *csvPath)

	X, y, err := loadCSV(*csvPath, featureCols, targetCol)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error cargando CSV:", err)
		os.Exit(1)
	}
	fmt.Printf("Filas: %d | Features (X): %d %v | Target (Y): %q\n",
		len(X), len(featureCols), featureCols, targetCol)
	fmt.Printf("Balance de clases -> alta peligrosidad(1): %.2f%%  |  baja(0): %.2f%%\n\n",
		100*positiveRate(y), 100*(1-positiveRate(y)))

	// El split usa la misma semilla siempre, así el conjunto de test es el
	// mismo tanto si entrenamos como si solo cargamos -> evaluación comparable.
	xTrain, xTest, yTrain, yTest := dataset.TrainTestSplit(X, y, *testRatio, *seed)
	fmt.Printf("Train: %d filas | Test: %d filas\n\n", len(xTrain), len(xTest))

	var (
		model *logreg.Model
		sc    *dataset.StandardScaler
	)

	if *loadOnly {
		// --- Cargar pesos guardados: predecir SIN reentrenar ---
		fmt.Printf("== Cargando modelo %q y scaler %q (sin reentrenar) ==\n", *modelPath, *scalerPath)
		if model, err = logreg.Load(*modelPath); err != nil {
			fmt.Fprintln(os.Stderr, "error cargando el modelo:", err)
			os.Exit(1)
		}
		if sc, err = dataset.LoadStandardScaler(*scalerPath); err != nil {
			fmt.Fprintln(os.Stderr, "error cargando el scaler:", err)
			os.Exit(1)
		}
		xTest = sc.Transform(xTest) // misma estandarización aprendida en el train
	} else {
		// --- Entrenar y guardar ---
		sc = dataset.NewStandardScaler()
		xTrain = sc.FitTransform(xTrain) // media/std aprendidos SOLO con el train
		xTest = sc.Transform(xTest)      // y reutilizados en test (sin leakage)

		fmt.Println("== Entrenamiento (verbose) ==")
		model = logreg.New(
			logreg.WithLearningRate(*lr),
			logreg.WithEpochs(*epochs),
			logreg.WithL2(*l2),
			logreg.WithThreshold(*threshold),
			logreg.WithWorkers(*workers),
			logreg.WithSeed(*seed),
			logreg.WithVerbose(true),
		)
		start := time.Now()
		if err := model.Fit(xTrain, yTrain); err != nil {
			fmt.Fprintln(os.Stderr, "error entrenando:", err)
			os.Exit(1)
		}
		fmt.Printf("\nEntrenado en %s\n", time.Since(start).Round(time.Millisecond))

		// --- Exportar el modelo y el scaler para reusarlos sin reentrenar ---
		if err := model.Save(*modelPath); err != nil {
			fmt.Fprintln(os.Stderr, "error guardando el modelo:", err)
			os.Exit(1)
		}
		if err := sc.Save(*scalerPath); err != nil {
			fmt.Fprintln(os.Stderr, "error guardando el scaler:", err)
			os.Exit(1)
		}
		fmt.Printf("Modelo guardado en %q | scaler en %q\n", *modelPath, *scalerPath)
	}

	// --- Validación / métricas en test ---
	pred := model.Predict(xTest)
	fmt.Println("\n== Métricas en TEST ==")
	fmt.Println(metrics.Report(yTest, pred))

	// Referencia: clasificador "tonto" que siempre predice la clase mayoritaria.
	fmt.Printf("\nBaseline (clase mayoritaria) accuracy: %.4f\n", baselineAccuracy(yTest))

	// --- Importancia de features (peso aprendido sobre datos estandarizados) ---
	fmt.Println("\n== Pesos aprendidos (datos estandarizados) ==")
	for i, w := range model.Weights() {
		fmt.Printf("  %-18s % .4f\n", featureCols[i], w)
	}
	fmt.Printf("  %-18s % .4f\n", "(bias)", model.Bias())
}

// loadCSV lee el CSV y construye X (solo las columnas de feats, en ese orden) e
// y (la columna target). Selecciona columnas por NOMBRE de cabecera, así el
// orden físico del archivo no importa.
func loadCSV(path string, feats []string, target string) (X [][]float64, y []float64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.ReuseRecord = true // menos allocations en archivos grandes

	header, err := r.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("leyendo cabecera: %w", err)
	}
	colIdx := map[string]int{}
	for i, name := range header {
		colIdx[strings.TrimSpace(name)] = i
	}

	// Resolvemos los índices de las columnas que nos interesan.
	featIdx := make([]int, len(feats))
	for i, name := range feats {
		j, ok := colIdx[name]
		if !ok {
			return nil, nil, fmt.Errorf("la columna feature %q no existe en el CSV", name)
		}
		featIdx[i] = j
	}
	tgtIdx, ok := colIdx[target]
	if !ok {
		return nil, nil, fmt.Errorf("la columna target %q no existe en el CSV", target)
	}

	line := 1
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		line++
		if err != nil {
			return nil, nil, fmt.Errorf("línea %d: %w", line, err)
		}

		row := make([]float64, len(featIdx))
		for i, j := range featIdx {
			v, perr := parseField(rec[j])
			if perr != nil {
				return nil, nil, fmt.Errorf("línea %d, columna %q: %w", line, feats[i], perr)
			}
			row[i] = v
		}
		yv, perr := parseField(rec[tgtIdx])
		if perr != nil {
			return nil, nil, fmt.Errorf("línea %d, columna %q: %w", line, target, perr)
		}

		X = append(X, row)
		y = append(y, yv)
	}
	if len(X) == 0 {
		return nil, nil, fmt.Errorf("el CSV no tiene filas de datos")
	}
	return X, y, nil
}

// parseField convierte un campo a float64. Acepta números y también booleanos
// ("true"/"false") que el dataset usa en is_night, domestic e is_high_risk.
func parseField(s string) (float64, error) {
	s = strings.TrimSpace(s)
	switch strings.ToLower(s) {
	case "true":
		return 1, nil
	case "false":
		return 0, nil
	}
	return strconv.ParseFloat(s, 64)
}

// positiveRate devuelve la fracción de etiquetas iguales a 1.
func positiveRate(y []float64) float64 {
	var pos int
	for _, v := range y {
		if v == 1 {
			pos++
		}
	}
	return float64(pos) / float64(len(y))
}

// baselineAccuracy es la exactitud de predecir siempre la clase mayoritaria.
func baselineAccuracy(y []float64) float64 {
	p := positiveRate(y)
	if p > 0.5 {
		return p
	}
	return 1 - p
}
