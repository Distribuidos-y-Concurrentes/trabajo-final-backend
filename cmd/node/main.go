// Command node es un NODO de cómputo del clúster de entrenamiento distribuido.
//
// Una sola imagen Docker: al arrancar se conecta al coordinador, recibe su rank y
// el reparto del dataset, carga SOLO su shard local, lo estandariza con el scaler
// compartido y, en cada ronda, calcula el gradiente parcial (en paralelo con
// goroutines, vía logreg.GradSums) y lo devuelve al coordinador por TCP. Escala con
// `docker compose up --scale node=N`: cada réplica se autoasigna un rank distinto.
package main

import (
	"encoding/json"
	"log"
	"net"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/crimedata"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/dataset"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/distrib"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/logreg"
)

func main() {
	coordAddr := env("COORDINATOR_ADDR", "coordinator:9100")
	dataPath := env("DATA_PATH", "/data/crimes_clean.csv")
	scalerPath := env("SCALER_PATH", "/data/crime_scaler.json")
	workers := envInt("WORKERS", runtime.NumCPU())

	host, _ := os.Hostname()
	log.Printf("[nodo %s] arrancando | coordinador=%s | workers=%d", host, coordAddr, workers)

	// El coordinador puede tardar en levantar: reintentamos la conexión.
	conn := dialWithRetry(coordAddr, 30, 2*time.Second)
	defer conn.Close()
	enc := json.NewEncoder(conn)
	dec := json.NewDecoder(conn)

	// 1) Registro: nos anunciamos y recibimos rank + reparto del dataset.
	if err := enc.Encode(distrib.RegisterReq{Hostname: host, Workers: workers}); err != nil {
		log.Fatalf("[nodo %s] error registrándose: %v", host, err)
	}
	var reg distrib.RegisterResp
	if err := dec.Decode(&reg); err != nil {
		log.Fatalf("[nodo %s] error recibiendo la asignación: %v", host, err)
	}
	start, end := distrib.ShardRange(reg.NRows, reg.World, reg.Rank)
	log.Printf("[nodo %s] rank=%d/%d | shard filas [%d,%d) (%d muestras)",
		host, reg.Rank, reg.World, start, end, end-start)

	// 2) Cargar SOLO mi shard y estandarizarlo con el scaler compartido (mismas
	//    medias/desviaciones que el resto de nodos => modelo consistente).
	X, y, err := crimedata.LoadRange(dataPath, start, end)
	if err != nil {
		log.Fatalf("[nodo %s] error cargando el shard: %v", host, err)
	}
	sc, err := dataset.LoadStandardScaler(scalerPath)
	if err != nil {
		log.Fatalf("[nodo %s] error cargando el scaler: %v", host, err)
	}
	X = sc.Transform(X)
	log.Printf("[nodo %s] shard listo: %d muestras x %d features", host, len(X), reg.NFeat)

	// El modelo local solo se usa para inyectar pesos y calcular el gradiente; los
	// workers de GradSums son el paralelismo INTRA-nodo.
	model := logreg.New(logreg.WithWorkers(workers))

	// 3) Rondas: recibir pesos globales -> gradiente parcial -> responder sumas.
	for {
		var req distrib.GradReq
		if err := dec.Decode(&req); err != nil {
			log.Fatalf("[nodo %s] error recibiendo la ronda: %v", host, err)
		}
		if req.Done {
			log.Printf("[nodo %s] entrenamiento finalizado, cerrando", host)
			return
		}
		model.SetParams(req.Weights, req.Bias)
		sumW, sumB, sumLoss, count := model.GradSums(X, y)
		if err := enc.Encode(distrib.GradResp{
			SumW: sumW, SumB: sumB, SumLoss: sumLoss, Count: count,
		}); err != nil {
			log.Fatalf("[nodo %s] error enviando el gradiente: %v", host, err)
		}
	}
}

// dialWithRetry intenta conectar al coordinador varias veces (en compose el nodo
// puede arrancar antes de que el coordinador esté escuchando).
func dialWithRetry(addr string, attempts int, wait time.Duration) net.Conn {
	for i := 0; i < attempts; i++ {
		conn, err := net.Dial("tcp", addr)
		if err == nil {
			return conn
		}
		log.Printf("coordinador %s no disponible (intento %d/%d): %v", addr, i+1, attempts, err)
		time.Sleep(wait)
	}
	log.Fatalf("no se pudo conectar al coordinador %s tras %d intentos", addr, attempts)
	return nil
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
