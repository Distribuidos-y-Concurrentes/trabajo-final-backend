# Urban Crime Predictor

Sistema distribuido de prediccion de crimenes urbanos construido en Go. Procesa el dataset de crimenes de Chicago (8.5 millones de registros) de forma concurrente para entrenar un modelo de regresion logistica binaria que identifica delitos de alta peligrosidad.

Trabajo final del curso CC65 Programacion Concurrente y Distribuida — UPC, Seccion 16661.

---

## Autores

| Nombre | Codigo |
|--------|--------|
| Daniella Alexandra Crysti Vargas Saldana | U202219211 |
| Jorge Sebastian Valdivia Peceros | U202212452 |
| Joaquin Eduardo Velarde Leyva | U202212510 |

Profesor: Carlos Alberto Jara Garcia

---

## Descripcion del problema

Las fuerzas del orden asignan patrullaje de manera reactiva ante delitos ya ocurridos, lo que genera distribucion ineficiente de recursos de seguridad.

Pregunta central que guia el sistema:

> En que zonas, dias y horas existe mayor probabilidad de ocurrencia de un delito de alta peligrosidad para optimizar el patrullaje preventivo?

## Dataset

| Campo | Detalle |
|-------|---------|
| Nombre | Chicago Crime Dataset (2019-2024) |
| Fuente | Chicago Data Portal / data.gov |
| Volumen | 8,567,753 registros |
| Licencia | Uso libre academico |

---

## Arquitectura

El sistema se divide en cuatro partes desplegadas con Docker Compose:

| Parte | Descripcion | Estado |
|-------|-------------|--------|
| A — Cargador concurrente | Lee el CSV en chunks paralelos, valida, deduplica y genera `crimes_clean.csv` | Completo |
| B — Entrenamiento distribuido | Parameter server TCP: coordinator + N nodos de computo de gradientes | Completo |
| C — Inferencia + API HTTP | Cluster de nodos de inferencia con balanceo least-connections y API REST | Completo |
| D — Visualizacion | Frontend React con mapas de calor y metricas del cluster | Pendiente |

Infraestructura: Redis (descubrimiento de nodos y balanceo de carga), MongoDB (persistencia historica de predicciones).

---

## Estructura del repositorio

```
trabajo-final-backend/
├── cmd/
│   ├── api/              # API HTTP REST (predict / health / metrics)
│   ├── cleaning/         # Cargador y limpiador concurrente del dataset
│   ├── coordinator/      # Coordinador del entrenamiento distribuido (parameter server)
│   ├── inference/        # Nodo de inferencia TCP con heartbeat a Redis
│   ├── node/             # Nodo de computo de gradientes (entrenamiento distribuido)
│   └── train_crimes/     # CLI de entrenamiento local con metricas
├── internal/
│   ├── balancer/         # Balanceador least-connections sobre Redis
│   ├── crimedata/        # Columnas del dataset y lectura de CSV
│   ├── dataset/          # TrainTestSplit y StandardScaler
│   ├── distrib/          # Protocolo TCP (entrenamiento e inferencia)
│   ├── logreg/           # Regresion logistica con gradiente paralelo
│   ├── metrics/          # Accuracy, Precision, Recall, F1, matriz de confusion
│   └── registry/         # Registro de nodos de inferencia en Redis (TTL)
├── docker-compose.distributed.yml   # Stack de entrenamiento distribuido
├── docker-compose.inference.yml     # Stack de inferencia: Redis + MongoDB + nodos + API
├── crime_model.json      # Pesos del modelo serializado (entrenamiento local)
├── crime_model_distributed.json     # Pesos del modelo serializado (entrenamiento distribuido)
├── crime_scaler.json     # Parametros del StandardScaler serializado
└── go.mod
```

---

## Entregable 1 — Limpieza y modelo local

### Limpieza concurrente (Parte A)

El cargador divide el archivo CSV en chunks de bytes y asigna cada chunk a una goroutine. Usa `sync.Map` para deduplicacion por ID sin contention y contadores `atomic` para el reporte final.

Resultados sobre el dataset completo con 4 workers:

| Metrica | Valor |
|---------|-------|
| Registros leidos | 8,567,753 |
| Registros validos | 8,470,795 (98.87%) |
| Tiempo de ejecucion | ~2 min 23 s |

Filtros aplicados: coordenadas fuera del rango de Chicago, distritos fuera del rango 1-31, duplicados por ID, fechas mal formateadas, filas incompletas.

### Modelo de ML (local)

Regresion logistica binaria con gradiente descendente paralelizado. El metodo `gradient` reparte el batch entre goroutines con buffers locales independientes y sin locks; se sincroniza con `sync.WaitGroup` al final de cada epoca.

- **Target (`is_high_risk`):** Assault, Battery, Robbery, Homicide, Weapons Violation, Criminal Sexual Assault, Arson
- **Features (9 variables):** `district`, `grid_lat`, `grid_lon`, `hour`, `weekday`, `month`, `is_night`, `location_encoded`, `domestic`

Metricas en test (split 80/20, seed=42):

| Metrica | Valor | Baseline (clase mayoritaria) |
|---------|-------|------------------------------|
| Accuracy | 0.7542 | 0.6894 |
| Precision | 0.6862 | — |
| Recall | 0.3842 | — |
| F1-score | 0.4926 | — |

---

## Entregable 2 — Entrenamiento distribuido y serving

### Entrenamiento distribuido (Parte B)

Parameter server sincrono sobre TCP. El coordinator espera a que se registren `WORLD_SIZE` nodos, les reparte el dataset en shards y por cada epoca:

1. Difunde los pesos globales a todos los nodos en paralelo.
2. Cada nodo calcula las sumas del gradiente sobre su shard local (con goroutines internas).
3. El coordinator recolecta las sumas de todos los nodos (barrera de sincronizacion), promedia globalmente `Σsum / Σcount`, aplica regularizacion L2 y actualiza los pesos.

Los datos nunca salen del nodo: por la red solo viajan pesos y gradientes.

```bash
# Levantar coordinator + 4 nodos de computo
docker compose -f docker-compose.distributed.yml up --build
```

El modelo resultante se guarda en `crime_model_distributed.json`.

### Cluster de inferencia (Parte C)

Cada nodo de inferencia (`cmd/inference`) carga el modelo y el scaler, atiende predicciones por TCP y se autorregistra en Redis con un TTL. Si el nodo muere, su clave expira y el balanceador deja de verlo automaticamente.

El balanceador (`internal/balancer`) vive embebido en la API: consulta Redis para obtener los nodos vivos, desempata con shuffle y elige el de menor carga efectiva (carga publicada en Redis + peticiones en vuelo locales).

### API HTTP (Parte C)

Servidor REST en `:8080` que actua como punto de entrada unico al cluster.

| Endpoint | Descripcion |
|----------|-------------|
| `POST /predict` | Recibe features en JSON, reenvía al nodo con menor carga, devuelve la prediccion |
| `GET /health` | Liveness probe |
| `GET /metrics` | Contadores de requests/errores y estado actual de cada nodo (carga en vuelo) |

**Ejemplo de prediccion:**

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"district":3,"hour":22,"is_night":1,"day_of_week":5,"weekday":5,"month":6,"grid_lat":41.85,"grid_lon":-87.65,"location_encoded":2,"domestic":0}}'
```

```json
{
  "prob": 0.7821,
  "is_high_risk": true,
  "node": "abc123def456:9200"
}
```

**Metricas del cluster:**

```bash
curl http://localhost:8080/metrics
```

```json
{
  "requests_total": 142,
  "errors_total": 3,
  "nodes": [
    { "addr": "abc123:9200", "load": 2 },
    { "addr": "def456:9200", "load": 0 }
  ],
  "nodes_error": ""
}
```

**Levantar el stack completo de inferencia:**

```bash
# Redis + MongoDB + 3 nodos de inferencia + API
docker compose -f docker-compose.inference.yml up --build --scale inference=3
```

---

## Ejecucion local (sin Docker)

### Requisitos

- Go 1.22 o superior
- `crimes.csv` del Chicago Data Portal ubicado en `cmd/cleaning/`

### Limpieza del dataset

```bash
cd cmd/cleaning
go run limpieza.go
```

Genera `crimes_clean.csv` en el mismo directorio.

### Entrenamiento local

```bash
go run cmd/train_crimes/main.go
```

Flags disponibles:

| Flag | Default | Descripcion |
|------|---------|-------------|
| `-csv` | `cmd/cleaning/crimes_clean.csv` | Ruta al CSV limpio |
| `-epochs` | 300 | Numero de epocas |
| `-lr` | 0.5 | Tasa de aprendizaje |
| `-l2` | 0.001 | Regularizacion L2 (lambda) |
| `-test` | 0.2 | Fraccion para test |
| `-threshold` | 0.5 | Umbral de decision |
| `-workers` | numCPU | Goroutines para el gradiente |
| `-seed` | 42 | Semilla para reproducibilidad |
| `-load` | false | Cargar modelo guardado y evaluar sin reentrenar |

Para evaluar el modelo serializado sin reentrenar:

```bash
go run cmd/train_crimes/main.go -load
```

---

## Guia de contribucion

Ver [CONTRIBUTING.md](CONTRIBUTING.md).
