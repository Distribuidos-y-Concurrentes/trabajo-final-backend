// Package distrib define el protocolo TCP entre el COORDINADOR del entrenamiento
// y los NODOS de cómputo (un parameter server síncrono). Los mensajes viajan como
// JSON delimitado por saltos de línea (encoding/json Encoder/Decoder sobre la
// conexión), de modo que por la red solo circulan pesos y gradientes —nunca los
// datos—: cada nodo entrena sobre su propio shard local del dataset.
//
// Secuencia sobre una conexión persistente (el nodo marca al coordinador):
//
//	nodo  --RegisterReq-->  coordinador
//	nodo  <-RegisterResp--  coordinador   (rank + reparto del dataset)
//	bucle por época:
//	  nodo  <-GradReq--   coordinador      (pesos globales actuales)
//	  nodo  --GradResp->  coordinador      (sumas del gradiente sobre su shard)
//	nodo  <-GradReq(Done)-- coordinador     (fin del entrenamiento)
package distrib

// RegisterReq lo envía el nodo al conectarse para anunciarse al coordinador.
type RegisterReq struct {
	Hostname string `json:"hostname"`
	Workers  int    `json:"workers"`
}

// RegisterResp es la respuesta del coordinador una vez registrados los WORLD_SIZE
// nodos: asigna el rank y comunica cómo se reparte el dataset.
type RegisterResp struct {
	Rank  int `json:"rank"`   // identificador 0..World-1 de este nodo
	World int `json:"world"`  // número total de nodos
	NRows int `json:"n_rows"` // filas de datos totales (para repartir en shards)
	NFeat int `json:"n_feat"` // número de features (longitud del vector de pesos)
}

// GradReq lo difunde el coordinador en cada ronda con los pesos globales actuales.
// Done=true indica fin del entrenamiento: el nodo debe terminar.
type GradReq struct {
	Epoch   int       `json:"epoch"`
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
	Done    bool      `json:"done"`
}

// GradResp es la respuesta de un nodo: las SUMAS (sin promediar) del gradiente y de
// la pérdida sobre su shard, más el nº de muestras procesadas. El coordinador
// agrega Σsum / Σcount entre todos los nodos.
type GradResp struct {
	SumW    []float64 `json:"sum_w"`
	SumB    float64   `json:"sum_b"`
	SumLoss float64   `json:"sum_loss"`
	Count   int       `json:"count"`
}

// ShardRange devuelve el rango [start,end) de filas asignado al nodo `rank` de un
// total de `world` particiones sobre `nRows` filas. El resto (nRows % world) se
// reparte entre los primeros nodos para que las particiones queden equilibradas.
func ShardRange(nRows, world, rank int) (start, end int) {
	base := nRows / world
	rem := nRows % world
	start = rank*base + min(rank, rem)
	end = start + base
	if rank < rem {
		end++
	}
	return start, end
}
