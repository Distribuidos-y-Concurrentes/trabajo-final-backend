// Package registry implementa el descubrimiento de nodos de inferencia sobre Redis.
//
// Cada nodo publica su dirección y su carga actual (peticiones en vuelo) bajo una
// clave con TTL. Mientras el nodo siga vivo refresca la clave con un heartbeat; si
// muere y deja de refrescarla, la clave EXPIRA y el balanceador deja de verlo. Así
// el balanceador puede repartir el trabajo al nodo con menos carga (least-connections)
// sin configuración estática de direcciones.
package registry

import (
	"context"
	"encoding/json"
	"time"

	"github.com/redis/go-redis/v9"
)

// KeyPrefix es el prefijo de las claves de nodos vivos en Redis (infer:nodes:<addr>).
const KeyPrefix = "infer:nodes:"

// Node es el estado publicado de un nodo de inferencia.
type Node struct {
	Addr string `json:"addr"` // dirección TCP donde atiende predicciones
	Load int64  `json:"load"` // peticiones en vuelo en este momento
}

// Publish escribe (o refresca) el estado del nodo con expiración ttl. Lo llaman el
// heartbeat (para mantener viva la clave) y el handler de peticiones (cuando cambia
// la carga). Escrituras concurrentes son seguras: gana la última (last-write-wins).
func Publish(ctx context.Context, rdb *redis.Client, n Node, ttl time.Duration) error {
	data, err := json.Marshal(n)
	if err != nil {
		return err
	}
	return rdb.Set(ctx, KeyPrefix+n.Addr, data, ttl).Err()
}

// Unregister borra la clave del nodo (apagado limpio, para no esperar al TTL).
func Unregister(ctx context.Context, rdb *redis.Client, addr string) error {
	return rdb.Del(ctx, KeyPrefix+addr).Err()
}

// List devuelve los nodos vivos (claves no expiradas). Lo usará el balanceador para
// elegir el nodo con menor carga. Recorre las claves con SCAN (no bloquea Redis).
func List(ctx context.Context, rdb *redis.Client) ([]Node, error) {
	var nodes []Node
	iter := rdb.Scan(ctx, 0, KeyPrefix+"*", 100).Iterator()
	for iter.Next(ctx) {
		val, err := rdb.Get(ctx, iter.Val()).Result()
		if err != nil {
			continue // pudo expirar entre el SCAN y el GET
		}
		var n Node
		if err := json.Unmarshal([]byte(val), &n); err == nil {
			nodes = append(nodes, n)
		}
	}
	return nodes, iter.Err()
}
