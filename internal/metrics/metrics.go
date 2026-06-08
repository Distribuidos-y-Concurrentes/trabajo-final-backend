// Package metrics contiene métricas de evaluación para clasificación binaria.
package metrics

import "fmt"

// Confusion es la matriz de confusión para clasificación binaria.
type Confusion struct {
	TP, TN, FP, FN int
}

// ConfusionMatrix calcula la matriz de confusión a partir de las etiquetas
// reales (yTrue) y las predichas (yPred), con clases 0.0 / 1.0.
func ConfusionMatrix(yTrue, yPred []float64) Confusion {
	var c Confusion
	for i := range yTrue {
		switch {
		case yTrue[i] == 1 && yPred[i] == 1:
			c.TP++
		case yTrue[i] == 0 && yPred[i] == 0:
			c.TN++
		case yTrue[i] == 0 && yPred[i] == 1:
			c.FP++
		default: // yTrue==1, yPred==0
			c.FN++
		}
	}
	return c
}

// Accuracy = (TP+TN) / total.
func Accuracy(yTrue, yPred []float64) float64 {
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return safeDiv(float64(correct), float64(len(yTrue)))
}

// Precision = TP / (TP+FP).
func Precision(yTrue, yPred []float64) float64 {
	c := ConfusionMatrix(yTrue, yPred)
	return safeDiv(float64(c.TP), float64(c.TP+c.FP))
}

// Recall = TP / (TP+FN).
func Recall(yTrue, yPred []float64) float64 {
	c := ConfusionMatrix(yTrue, yPred)
	return safeDiv(float64(c.TP), float64(c.TP+c.FN))
}

// F1 es la media armónica de precisión y recall.
func F1(yTrue, yPred []float64) float64 {
	p := Precision(yTrue, yPred)
	r := Recall(yTrue, yPred)
	return safeDiv(2*p*r, p+r)
}

// Report imprime un resumen legible de todas las métricas.
func Report(yTrue, yPred []float64) string {
	c := ConfusionMatrix(yTrue, yPred)
	return fmt.Sprintf(
		"Accuracy : %.4f\nPrecision: %.4f\nRecall   : %.4f\nF1-score : %.4f\n"+
			"Matriz de confusión -> TP:%d TN:%d FP:%d FN:%d",
		Accuracy(yTrue, yPred), Precision(yTrue, yPred),
		Recall(yTrue, yPred), F1(yTrue, yPred),
		c.TP, c.TN, c.FP, c.FN,
	)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}
