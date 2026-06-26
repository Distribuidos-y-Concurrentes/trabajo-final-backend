// Command inference es un NODO DE INFERENCIA del clúster. Carga el modelo entrenado
// (de forma distribuida) y su scaler, y atiende predicciones por TCP —un JSON por
// línea, el mismo estilo que el clúster de entrenamiento—. Se autorregistra en
// Redis con un heartbeat TTL para que el balanceador lo descubra y le reparta carga
// según las peticiones en vuelo. Una sola imagen: escala con réplicas.
package main

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/crimedata"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/dataset"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/distrib"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/logreg"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/registry"
)

// server agrupa el estado del nodo de inferencia.
type server struct {
	model     *logreg.Model
	scaler    *dataset.StandardScaler
	rdb       *redis.Client
	advertise string        // dirección con la que se anuncia en Redis
	ttl       time.Duration // expiración del registro
	inFlight  atomic.Int64  // peticiones en vuelo (carga actual)
}

func main() {
	listenAddr := env("LISTEN_ADDR", ":9200")
	modelPath := env("MODEL_PATH", "/data/crime_model_distributed.json")
	scalerPath := env("SCALER_PATH", "/data/crime_scaler.json")
	redisAddr := env("REDIS_ADDR", "redis:6379")
	ttl := time.Duration(envInt("HEARTBEAT_TTL", 10)) * time.Second

	model, err := logreg.Load(modelPath)
	if err != nil {
		log.Fatalf("[infer] error cargando el modelo %q: %v", modelPath, err)
	}
	scaler, err := dataset.LoadStandardScaler(scalerPath)
	if err != nil {
		log.Fatalf("[infer] error cargando el scaler %q: %v", scalerPath, err)
	}

	// Dirección anunciada: por defecto hostname:puerto (en Docker el hostname es el
	// ID del contenedor, alcanzable por el resto de la red). Sobrescribible por env.
	host, _ := os.Hostname()
	_, port, _ := net.SplitHostPort(listenAddr)
	advertise := env("ADVERTISE_ADDR", net.JoinHostPort(host, port))

	srv := &server{
		model:     model,
		scaler:    scaler,
		rdb:       redis.NewClient(&redis.Options{Addr: redisAddr}),
		advertise: advertise,
		ttl:       ttl,
	}

	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("[infer] no se pudo escuchar en %s: %v", listenAddr, err)
	}
	defer ln.Close()
	log.Printf("[infer %s] sirviendo predicciones en %s | redis=%s | modelo=%s",
		advertise, listenAddr, redisAddr, modelPath)

	go srv.heartbeat()       // mantiene vivo el registro en Redis
	go srv.handleSignals(ln) // apagado limpio: se desregistra

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("[infer %s] accept terminó: %v", advertise, err)
			return
		}
		go srv.handle(conn)
	}
}

// heartbeat publica el estado del nodo al arrancar y luego cada ttl/2, de modo que
// la clave en Redis nunca expire mientras el nodo siga vivo.
func (s *server) heartbeat() {
	ctx := context.Background()
	s.publish(ctx)
	t := time.NewTicker(s.ttl / 2)
	defer t.Stop()
	for range t.C {
		s.publish(ctx)
	}
}

// publish escribe la carga actual del nodo en Redis (best-effort; si Redis no está
// disponible se registra el error pero el nodo sigue sirviendo predicciones).
func (s *server) publish(ctx context.Context) {
	n := registry.Node{Addr: s.advertise, Load: s.inFlight.Load()}
	if err := registry.Publish(ctx, s.rdb, n, s.ttl); err != nil {
		log.Printf("[infer %s] no se pudo registrar en Redis: %v", s.advertise, err)
	}
}

// handle atiende una conexión: lee peticiones de predicción (un JSON por línea) y
// responde otra línea JSON. Cuenta las peticiones en vuelo para el balanceo.
func (s *server) handle(conn net.Conn) {
	defer conn.Close()
	dec := json.NewDecoder(conn)
	enc := json.NewEncoder(conn)
	for {
		var req distrib.PredictReq
		if err := dec.Decode(&req); err != nil {
			return // conexión cerrada o JSON inválido
		}

		// La carga en vuelo se cuenta de forma atómica; el heartbeat (único escritor
		// de Redis) la publica periódicamente. NO publicamos aquí por petición para
		// evitar escrituras concurrentes que se reordenen y dejen una carga "pegada".
		s.inFlight.Add(1)
		resp := s.predict(req)
		_ = enc.Encode(resp)
		s.inFlight.Add(-1)
	}
}

// predict estandariza la entrada con el scaler y devuelve la probabilidad y la clase.
func (s *server) predict(req distrib.PredictReq) distrib.PredictResp {
	row := make([]float64, len(crimedata.FeatureCols))
	for i, f := range crimedata.FeatureCols {
		v, ok := req.Features[f]
		if !ok {
			return distrib.PredictResp{Node: s.advertise, Error: "falta la feature " + f}
		}
		row[i] = v
	}
	x := s.scaler.Transform([][]float64{row})
	prob := s.model.PredictProba(x)[0]
	cls := s.model.Predict(x)[0] == 1
	return distrib.PredictResp{Prob: prob, IsHighRisk: cls, Node: s.advertise}
}

// handleSignals desregistra el nodo de Redis ante SIGINT/SIGTERM (apagado limpio).
func (s *server) handleSignals(ln net.Listener) {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	<-ch
	log.Printf("[infer %s] apagando, desregistrando de Redis", s.advertise)
	_ = registry.Unregister(context.Background(), s.rdb, s.advertise)
	ln.Close()
	os.Exit(0)
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
