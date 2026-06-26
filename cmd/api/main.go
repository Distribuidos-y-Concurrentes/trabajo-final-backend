// Command api es la puerta de entrada HTTP al clúster de inferencia.
// Expone tres endpoints:
//
//	POST /predict  — recibe features en JSON y devuelve la predicción del modelo
//	GET  /health   — liveness probe (siempre 200 OK si el proceso está vivo)
//	GET  /metrics  — estadísticas de runtime: requests, errores, nodos activos
//
// El balanceo de carga hacia los nodos de inferencia está completamente encapsulado
// en internal/balancer: la API solo llama a bal.Predict y no sabe nada del clúster.
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/balancer"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/distrib"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/registry"
)

type apiServer struct {
	bal      balancer.Predictor
	rdb      *redis.Client
	requests atomic.Int64
	errors   atomic.Int64
}

func main() {
	listenAddr := env("LISTEN_ADDR", ":8080")
	redisAddr := env("REDIS_ADDR", "redis:6379")

	rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
	bal := balancer.New(rdb)

	srv := &apiServer{bal: bal, rdb: rdb}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", srv.handlePredict)
	mux.HandleFunc("GET /health", srv.handleHealth)
	mux.HandleFunc("GET /metrics", srv.handleMetrics)

	log.Printf("[api] escuchando en %s | redis=%s", listenAddr, redisAddr)
	if err := http.ListenAndServe(listenAddr, mux); err != nil {
		log.Fatalf("[api] error fatal: %v", err)
	}
}

// handlePredict decodifica la petición, la reenvía al clúster vía el balanceador
// y devuelve la respuesta. Los errores de infraestructura se mapean a 503.
func (s *apiServer) handlePredict(w http.ResponseWriter, r *http.Request) {
	s.requests.Add(1)

	var req distrib.PredictReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.errors.Add(1)
		writeJSON(w, http.StatusBadRequest, distrib.PredictResp{Error: "JSON inválido: " + err.Error()})
		return
	}

	resp, err := s.bal.Predict(r.Context(), req)
	if err != nil {
		s.errors.Add(1)
		writeJSON(w, http.StatusServiceUnavailable, distrib.PredictResp{Error: err.Error()})
		return
	}

	if resp.Error != "" {
		s.errors.Add(1)
		writeJSON(w, http.StatusUnprocessableEntity, resp)
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

func (s *apiServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleMetrics expone métricas de runtime: contadores de la API y estado
// actual de cada nodo de inferencia (dirección + peticiones en vuelo).
func (s *apiServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()

	nodes, err := registry.List(ctx, s.rdb)
	nodesErr := ""
	if err != nil {
		nodesErr = err.Error()
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"requests_total": s.requests.Load(),
		"errors_total":   s.errors.Load(),
		"nodes":          nodes,
		"nodes_error":    nodesErr,
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
