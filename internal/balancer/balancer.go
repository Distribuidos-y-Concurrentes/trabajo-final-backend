// Package balancer reparte peticiones de inferencia entre los nodos del clúster
// usando least-connections. Está pensado para EMBEBERSE en la API: se crea una vez
// con New y se invoca Predict en cada petición. El descubrimiento de nodos (vía
// Redis) y el balanceo quedan encapsulados, así la API no sabe nada del clúster.
//
// Uso típico desde la API:
//
//	rdb := redis.NewClient(&redis.Options{Addr: "redis:6379"})
//	bal := balancer.New(rdb)
//	// dentro de un handler HTTP:
//	resp, err := bal.Predict(r.Context(), distrib.PredictReq{Features: feats})
package balancer

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"sort"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/distrib"
	"github.com/Distribuidos-y-Concurrentes/trabajo-final-backend/internal/registry"
)

// Predictor es la abstracción que la API puede referenciar (facilita inyección de
// dependencias y mocks en tests). *Balancer la implementa.
type Predictor interface {
	Predict(ctx context.Context, req distrib.PredictReq) (distrib.PredictResp, error)
}

// comprobación en tiempo de compilación de que *Balancer cumple Predictor.
var _ Predictor = (*Balancer)(nil)

// Balancer descubre los nodos vivos en Redis y reenvía cada petición al de menor
// carga. Es seguro para uso concurrente: una sola instancia sirve a toda la API.
type Balancer struct {
	rdb         *redis.Client
	dialTimeout time.Duration

	mu      sync.Mutex     // protege pending
	pending map[string]int // peticiones que este balanceador tiene en vuelo por nodo
}

// Option configura el Balancer (functional options, igual estilo que internal/logreg).
type Option func(*Balancer)

// WithDialTimeout fija el timeout de conexión/lectura hacia los nodos (def. 2s).
func WithDialTimeout(d time.Duration) Option {
	return func(b *Balancer) { b.dialTimeout = d }
}

// New crea un Balancer que usa rdb para descubrir los nodos de inferencia.
func New(rdb *redis.Client, opts ...Option) *Balancer {
	b := &Balancer{
		rdb:         rdb,
		dialTimeout: 2 * time.Second,
		pending:     map[string]int{},
	}
	for _, o := range opts {
		o(b)
	}
	return b
}

// Predict elige el nodo de inferencia con menos carga efectiva y le reenvía la
// petición, reintentando con los siguientes si alguno falla.
//
// Devuelve error SOLO ante fallos de infraestructura (Redis caído, ningún nodo
// vivo, o todos los nodos fallan). Los errores de validación del modelo (p. ej. una
// feature faltante) viajan dentro de PredictResp.Error con error == nil.
func (b *Balancer) Predict(ctx context.Context, req distrib.PredictReq) (distrib.PredictResp, error) {
	nodes, err := registry.List(ctx, b.rdb)
	if err != nil {
		return distrib.PredictResp{}, fmt.Errorf("balancer: consultando Redis: %w", err)
	}
	if len(nodes) == 0 {
		return distrib.PredictResp{}, fmt.Errorf("balancer: no hay nodos de inferencia disponibles")
	}

	// Barajar rompe los empates de carga de forma uniforme; luego ordenar por carga
	// EFECTIVA = la publicada en Redis + lo que este balanceador ya despachó.
	rand.Shuffle(len(nodes), func(i, j int) { nodes[i], nodes[j] = nodes[j], nodes[i] })
	sort.SliceStable(nodes, func(i, j int) bool {
		return b.effLoad(nodes[i]) < b.effLoad(nodes[j])
	})

	var lastErr error
	for _, n := range nodes {
		b.addPending(n.Addr, +1)
		resp, err := b.forward(ctx, n.Addr, req)
		b.addPending(n.Addr, -1)
		if err == nil {
			return resp, nil
		}
		lastErr = err
	}
	return distrib.PredictResp{}, fmt.Errorf("balancer: todos los nodos fallaron: %w", lastErr)
}

// effLoad es la carga estimada de un nodo: la que publicó en Redis más las
// peticiones que este balanceador le tiene en vuelo (evita mandar una ráfaga entera
// al mismo nodo antes de que su carga se refresque en Redis).
func (b *Balancer) effLoad(n registry.Node) int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return n.Load + int64(b.pending[n.Addr])
}

func (b *Balancer) addPending(addr string, d int) {
	b.mu.Lock()
	b.pending[addr] += d
	if b.pending[addr] == 0 {
		delete(b.pending, addr)
	}
	b.mu.Unlock()
}

// forward abre una conexión TCP al nodo, le envía la petición y devuelve su
// respuesta. Usa deadlines para no quedarse colgado si el nodo no responde.
func (b *Balancer) forward(ctx context.Context, addr string, req distrib.PredictReq) (distrib.PredictResp, error) {
	d := net.Dialer{Timeout: b.dialTimeout}
	conn, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return distrib.PredictResp{}, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(b.dialTimeout))

	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return distrib.PredictResp{}, err
	}
	var resp distrib.PredictResp
	if err := json.NewDecoder(conn).Decode(&resp); err != nil {
		return distrib.PredictResp{}, err
	}
	return resp, nil
}
