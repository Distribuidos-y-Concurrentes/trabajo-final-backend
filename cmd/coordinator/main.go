// Command coordinator orquesta el ENTRENAMIENTO DISTRIBUIDO (parameter server
// síncrono). Hace de coordinador del clúster: espera a que se registren WORLD_SIZE
// nodos, reparte el dataset entre ellos y, en cada época, difunde los pesos
// globales, recolecta los gradientes parciales de TODOS los nodos (barrera de
// sincronización), los promedia, añade la regularización L2 y actualiza los pesos.
// Al terminar guarda el modelo reutilizando logreg.Save.
package main

import (
	"encoding/json"
	"log"
	"net"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/crimedata"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/distrib"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/logreg"
)

// nodeConn agrupa la conexión persistente con un nodo y sus codificadores JSON.
type nodeConn struct {
	rank int
	conn net.Conn
	enc  *json.Encoder
	dec  *json.Decoder
}

func main() {
	listenAddr := env("LISTEN_ADDR", ":9100")
	world := envInt("WORLD_SIZE", 4)
	dataPath := env("DATA_PATH", "/data/crimes_clean.csv")
	modelOut := env("MODEL_OUT", "/out/crime_model_distributed.json")
	epochs := envInt("EPOCHS", 300)
	lr := envFloat("LR", 0.5)
	l2 := envFloat("L2", 0.001)

	nFeat := len(crimedata.FeatureCols)

	log.Printf("[coord] contando filas de %s ...", dataPath)
	nRows, err := crimedata.CountRows(dataPath)
	if err != nil {
		log.Fatalf("[coord] error contando filas: %v", err)
	}
	log.Printf("[coord] %d filas | %d features | esperando %d nodos en %s",
		nRows, nFeat, world, listenAddr)

	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("[coord] no se pudo escuchar en %s: %v", listenAddr, err)
	}
	defer ln.Close()

	// 1) Esperar a que se registren WORLD_SIZE nodos.
	nodes := make([]*nodeConn, 0, world)
	for len(nodes) < world {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("[coord] error aceptando conexión: %v", err)
			continue
		}
		dec := json.NewDecoder(conn)
		enc := json.NewEncoder(conn)
		var req distrib.RegisterReq
		if err := dec.Decode(&req); err != nil {
			log.Printf("[coord] registro inválido, descartando conexión: %v", err)
			conn.Close()
			continue
		}
		rank := len(nodes)
		nodes = append(nodes, &nodeConn{rank: rank, conn: conn, enc: enc, dec: dec})
		log.Printf("[coord] nodo registrado rank=%d host=%s workers=%d (%d/%d)",
			rank, req.Hostname, req.Workers, len(nodes), world)
	}

	// 2) Asignar rank y reparto del dataset a cada nodo.
	for _, nd := range nodes {
		if err := nd.enc.Encode(distrib.RegisterResp{
			Rank: nd.rank, World: world, NRows: nRows, NFeat: nFeat,
		}); err != nil {
			log.Fatalf("[coord] error asignando al nodo %d: %v", nd.rank, err)
		}
	}

	// 3) Descenso de gradiente DISTRIBUIDO (full-batch por época, como el entrenador
	//    mono-nodo por defecto). Pesos globales: único escritor = este bucle.
	weights := make([]float64, nFeat)
	var bias float64

	log.Printf("[coord] iniciando entrenamiento: epochs=%d lr=%g l2=%g", epochs, lr, l2)
	startT := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		sumW := make([]float64, nFeat)
		var sumB, sumLoss float64
		var totalCount int

		// Difundir pesos y recolectar gradientes parciales en paralelo. wg.Wait()
		// es la barrera de sincronización entre nodos de cada época.
		var wg sync.WaitGroup
		var mu sync.Mutex
		errs := make([]error, len(nodes))
		for i, nd := range nodes {
			wg.Add(1)
			go func(i int, nd *nodeConn) {
				defer wg.Done()
				if err := nd.enc.Encode(distrib.GradReq{
					Epoch: epoch, Weights: weights, Bias: bias,
				}); err != nil {
					errs[i] = err
					return
				}
				var resp distrib.GradResp
				if err := nd.dec.Decode(&resp); err != nil {
					errs[i] = err
					return
				}
				mu.Lock()
				for j := range sumW {
					sumW[j] += resp.SumW[j]
				}
				sumB += resp.SumB
				sumLoss += resp.SumLoss
				totalCount += resp.Count
				mu.Unlock()
			}(i, nd)
		}
		wg.Wait()
		for i, e := range errs {
			if e != nil {
				log.Fatalf("[coord] nodo %d falló en la época %d: %v", i, epoch, e)
			}
		}
		if totalCount == 0 {
			log.Fatalf("[coord] época %d: 0 muestras agregadas", epoch)
		}

		// Promedio GLOBAL (Σsum / Σcount) + regularización L2 (una sola vez, sobre
		// los pesos globales) + paso de gradiente descendente.
		invN := 1.0 / float64(totalCount)
		for j := range weights {
			grad := sumW[j]*invN + l2*invN*weights[j]
			weights[j] -= lr * grad
		}
		bias -= lr * sumB * invN

		if epoch%max(1, epochs/10) == 0 || epoch == epochs-1 {
			loss := sumLoss*invN + l2Penalty(l2, invN, weights)
			log.Printf("[coord] época %4d/%d  loss=%.6f  (muestras=%d)",
				epoch+1, epochs, loss, totalCount)
		}
	}
	log.Printf("[coord] entrenamiento completado en %s", time.Since(startT).Round(time.Millisecond))

	// 4) Señal de fin a los nodos.
	for _, nd := range nodes {
		_ = nd.enc.Encode(distrib.GradReq{Done: true})
		nd.conn.Close()
	}

	// 5) Persistir el modelo entrenado (reutiliza logreg.SetParams + Save).
	model := logreg.New()
	model.SetParams(weights, bias)
	if err := model.Save(modelOut); err != nil {
		log.Fatalf("[coord] error guardando el modelo: %v", err)
	}
	log.Printf("[coord] modelo guardado en %s", modelOut)
	log.Printf("[coord] pesos=%v  bias=%.4f", weights, bias)
}

// l2Penalty replica el término de regularización que logreg añade a la pérdida,
// solo para el log informativo (la pérdida no afecta a la actualización).
func l2Penalty(l2, invN float64, w []float64) float64 {
	if l2 == 0 {
		return 0
	}
	var s float64
	for _, v := range w {
		s += v * v
	}
	return 0.5 * l2 * invN * s
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

func envFloat(k string, def float64) float64 {
	if v := os.Getenv(k); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
