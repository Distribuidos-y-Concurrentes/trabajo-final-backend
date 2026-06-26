package distrib

// PredictReq es la petición de inferencia que recibe un nodo por TCP (un JSON por
// línea). Features mapea nombre de feature -> valor; deben estar TODAS las que el
// modelo espera (ver crimedata.FeatureCols), p. ej. district, hour, is_night, ...
type PredictReq struct {
	Features map[string]float64 `json:"features"`
}

// PredictResp es la respuesta del nodo. Si Error != "", la predicción no se realizó.
// Node identifica qué nodo atendió la petición (útil para depurar el balanceo).
type PredictResp struct {
	Prob       float64 `json:"prob"`         // P(y=1 | x)
	IsHighRisk bool    `json:"is_high_risk"` // clase según el umbral del modelo
	Node       string  `json:"node,omitempty"`
	Error      string  `json:"error,omitempty"`
}
