// Package crimedata centraliza la especificación de features del dataset de
// crímenes y la carga del CSV por RANGOS de filas. Lo usa el entrenamiento
// distribuido: cada nodo carga únicamente su shard [start,end) en lugar de leer
// los más de 8 millones de registros completos en memoria.
//
// FeatureCols mantiene el MISMO orden que el entrenador mono-nodo
// (cmd/train_crimes) para que el scaler y los pesos sean compatibles.
package crimedata

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// FeatureCols son las columnas de X, en orden. Debe coincidir con featureCols de
// cmd/train_crimes y con el orden usado al ajustar el StandardScaler.
var FeatureCols = []string{
	"district",
	"grid_lat",
	"grid_lon",
	"hour",
	"weekday",
	"month",
	"is_night",
	"location_encoded",
	"domestic",
}

// TargetCol es la variable a predecir (Y).
const TargetCol = "is_high_risk"

// CountRows cuenta las filas de datos del CSV (sin contar la cabecera). El
// coordinador la llama UNA sola vez para repartir el dataset en shards.
func CountRows(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024) // líneas largas
	n := 0
	for sc.Scan() {
		n++
	}
	if err := sc.Err(); err != nil {
		return 0, err
	}
	if n == 0 {
		return 0, fmt.Errorf("crimedata: %q está vacío", path)
	}
	return n - 1, nil // descontamos la cabecera
}

// LoadRange carga las filas de datos cuyo índice (empezando en 0 tras la cabecera)
// cae en [start,end), construyendo X (solo FeatureCols, en orden) e y (TargetCol).
// Recorre el archivo en streaming y solo materializa su shard, así un nodo nunca
// guarda más muestras de las que le tocan.
func LoadRange(path string, start, end int) (X [][]float64, y []float64, err error) {
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

	featIdx := make([]int, len(FeatureCols))
	for i, name := range FeatureCols {
		j, ok := colIdx[name]
		if !ok {
			return nil, nil, fmt.Errorf("la columna feature %q no existe en el CSV", name)
		}
		featIdx[i] = j
	}
	tgtIdx, ok := colIdx[TargetCol]
	if !ok {
		return nil, nil, fmt.Errorf("la columna target %q no existe en el CSV", TargetCol)
	}

	idx := 0 // índice de fila de DATOS (0 = primera fila tras la cabecera)
	for {
		rec, rerr := r.Read()
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return nil, nil, fmt.Errorf("fila %d: %w", idx, rerr)
		}
		if idx >= end {
			break // ya pasamos el final de nuestro shard: paramos de leer
		}
		if idx < start {
			idx++
			continue // aún no llegamos al inicio de nuestro shard
		}

		row := make([]float64, len(featIdx))
		for i, j := range featIdx {
			v, perr := parseField(rec[j])
			if perr != nil {
				return nil, nil, fmt.Errorf("fila %d, columna %q: %w", idx, FeatureCols[i], perr)
			}
			row[i] = v
		}
		yv, perr := parseField(rec[tgtIdx])
		if perr != nil {
			return nil, nil, fmt.Errorf("fila %d, columna %q: %w", idx, TargetCol, perr)
		}

		X = append(X, row)
		y = append(y, yv)
		idx++
	}
	return X, y, nil
}

// parseField convierte un campo a float64. Acepta números y booleanos
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
