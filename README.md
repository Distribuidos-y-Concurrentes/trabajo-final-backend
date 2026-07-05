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

| Parte | Componente | Descripcion | Estado |
|-------|-----------|-------------|--------|
| A | `cmd/cleaning` | Cargador concurrente: divide el CSV en chunks, valida, deduplica y genera `crimes_clean.csv` | Completo |
| B | `cmd/coordinator` + `cmd/node` | Parameter server TCP sincrono: coordinator + N nodos de computo de gradientes | Completo |
| C | `cmd/inference` + `cmd/api` | Cluster de nodos de inferencia con balanceo least-connections y API REST | Completo |
| D | Frontend React | Mapas de calor y metricas del cluster | Pendiente |

Infraestructura: Redis (descubrimiento de nodos y balanceo de carga), MongoDB (persistencia historica de predicciones).

### Diagrama de comunicacion

```
[Cliente HTTP]
      |
      v
[API :8080]  --- Redis (service discovery) --->  [Nodo inferencia :9200]
                                                  [Nodo inferencia :9200]
                                                  [Nodo inferencia :9200]

[Coordinador :9100] --TCP--> [Nodo computo rank=0]
                    --TCP--> [Nodo computo rank=1]
                    --TCP--> [Nodo computo rank=2]
                    --TCP--> [Nodo computo rank=3]
```

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
│   ├── crimedata/        # Columnas del dataset y lectura de CSV por rangos
│   ├── dataset/          # TrainTestSplit y StandardScaler
│   ├── distrib/          # Protocolo TCP (entrenamiento e inferencia)
│   ├── logreg/           # Regresion logistica con gradiente paralelo
│   ├── metrics/          # Accuracy, Precision, Recall, F1, matriz de confusion
│   └── registry/         # Registro de nodos de inferencia en Redis (TTL)
├── docker-compose.distributed.yml   # Stack de entrenamiento distribuido
├── docker-compose.inference.yml     # Stack de inferencia: Redis + MongoDB + nodos + API
├── crime_model.json                 # Pesos del modelo (entrenamiento local)
├── crime_model_distributed.json     # Pesos del modelo (entrenamiento distribuido)
├── crime_scaler.json                # Parametros del StandardScaler serializado
└── go.mod
```

---

## API HTTP

El servidor REST (`cmd/api`) escucha en `:8080` y actua como punto de entrada unico al cluster. El balanceo de carga esta embebido en el mismo proceso (`internal/balancer`): consulta Redis para descubrir los nodos vivos y reenvía cada peticion al de menor carga efectiva por TCP.

### Endpoints

| Metodo | Ruta | Descripcion | Exito | Errores |
|--------|------|-------------|-------|---------|
| `POST` | `/predict` | Recibe features en JSON, elige el nodo con menor carga (least-connections), reenvía por TCP y devuelve probabilidad + clase | `200` | `400` JSON invalido, `422` feature faltante, `503` sin nodos |
| `GET` | `/health` | Liveness probe | `200` | — |
| `GET` | `/metrics` | Contadores de requests/errores y estado de cada nodo (carga en vuelo) | `200` | — |

### `POST /predict`

**Request:**

```json
{
  "features": {
    "district":          3,
    "grid_lat":          41.85,
    "grid_lon":          -87.65,
    "hour":              22,
    "weekday":           5,
    "month":             6,
    "is_night":          1,
    "location_encoded":  0,
    "domestic":          0
  }
}
```

| Feature | Tipo | Descripcion |
|---------|------|-------------|
| `district` | int (1-31) | Distrito policial de Chicago |
| `grid_lat` | float | Latitud redondeada a grilla de 0.004 grados |
| `grid_lon` | float | Longitud redondeada a grilla de 0.004 grados |
| `hour` | int (0-23) | Hora del dia del incidente |
| `weekday` | int (0-6) | Dia de la semana (0=domingo) |
| `month` | int (1-12) | Mes del ano |
| `is_night` | 0 o 1 | 1 si la hora es >= 20 o < 6 |
| `location_encoded` | int (0-8) | Tipo de lugar codificado numericamente |
| `domestic` | 0 o 1 | 1 si el incidente ocurrio en contexto domestico |

**Response exitosa (`200`):**

```json
{
  "prob": 0.7821,
  "is_high_risk": true,
  "node": "abc123def456:9200"
}
```

| Campo | Descripcion |
|-------|-------------|
| `prob` | Probabilidad P(y=1 \| x) calculada por el modelo logistico |
| `is_high_risk` | Clase predicha segun el umbral configurado (default 0.5) |
| `node` | Nodo de inferencia que atendio la peticion |

### `GET /metrics`

```json
{
  "requests_total": 142,
  "errors_total":   3,
  "nodes": [
    { "addr": "abc123:9200", "load": 2 },
    { "addr": "def456:9200", "load": 0 }
  ],
  "nodes_error": ""
}
```

### Variables de entorno — API

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8080` | Direccion de escucha del servidor HTTP |
| `REDIS_ADDR` | `redis:6379` | Direccion del servidor Redis |

---

## Almacenamiento

### Redis — Service discovery

Cada nodo de inferencia publica su estado bajo la clave `infer:nodes:<addr>` con TTL configurable. El balanceador escanea esas claves para obtener los nodos vivos y su carga actual.

```
infer:nodes:<host:puerto>  ->  { "addr": "...", "load": N }   TTL: 10s
```

Ciclo de vida: el nodo renueva la clave cada `TTL/2` segundos con un heartbeat. Si muere sin desregistrarse, la clave expira y el balanceador deja de verlo automaticamente. Ante SIGINT/SIGTERM el nodo llama a `registry.Unregister` para borrarla de inmediato.

### MongoDB — Historico de predicciones

Provisionado en el stack de inferencia para registrar el historico de predicciones. Base de datos `crimes`, coleccion `predictions`.

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `timestamp` | Date | Momento de la prediccion |
| `features` | Object | Mapa de features de la peticion |
| `prob` | Double | Probabilidad devuelta por el modelo |
| `is_high_risk` | Boolean | Clase predicha |
| `node` | String | Nodo que proceso la prediccion |

---

## Entregable 1 — Limpieza y modelo local

### Limpieza concurrente (Parte A)

El cargador (`cmd/cleaning/limpieza.go`) divide el archivo CSV en chunks de bytes y asigna cada chunk a una goroutine. Usa `sync.Map` para deduplicacion por ID sin contention y contadores `atomic` para el reporte final.

Resultados sobre el dataset completo con 4 workers:

| Metrica | Valor |
|---------|-------|
| Registros leidos | 8,567,753 |
| Registros validos | 8,470,795 (98.87%) |
| Tiempo de ejecucion | ~2 min 23 s |

Filtros aplicados: coordenadas fuera del rango de Chicago, distritos fuera del rango 1-31, duplicados por ID, fechas mal formateadas, filas incompletas.

### Modelo local (Parte A)

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

Pesos aprendidos sobre datos estandarizados:

| Feature | Peso |
|---------|------|
| `is_night` | +0.3821 |
| `hour` | +0.2134 |
| `location_encoded` | +0.1562 |
| `grid_lon` | +0.0891 |
| `district` | +0.0423 |
| `weekday` | -0.0312 |
| `month` | -0.0089 |
| `grid_lat` | -0.1187 |
| `domestic` | -0.4203 |

---

## Entregable 2 — Entrenamiento distribuido y serving

### Entrenamiento distribuido (Parte B)

Parameter server sincrono sobre TCP. El coordinator espera a que se registren `WORLD_SIZE` nodos, les reparte el dataset en shards y por cada epoca:

1. Difunde los pesos globales a todos los nodos en paralelo.
2. Cada nodo calcula las sumas del gradiente sobre su shard local (con goroutines internas).
3. El coordinator recolecta las sumas de todos los nodos (barrera `sync.WaitGroup`), promedia globalmente `Σsum / Σcount`, aplica regularizacion L2 y actualiza los pesos.

Los datos nunca salen del nodo: por la red solo viajan pesos y gradientes.

**Protocolo TCP (`internal/distrib`):**

```
nodo  --RegisterReq-->   coordinator       hostname + num_workers
nodo  <-RegisterResp--   coordinator       rank, world, n_rows, n_features

por cada epoca:
  nodo  <-GradReq--      coordinator       pesos globales actuales
  nodo  --GradResp->     coordinator       sum_w, sum_b, sum_loss, count

nodo  <-GradReq(Done)--  coordinator       senal de fin
```

```bash
docker compose -f docker-compose.distributed.yml up --build --scale node=4
```

Variables de entorno del coordinator:

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `WORLD_SIZE` | `4` | Numero de nodos (debe coincidir con `--scale node=N`) |
| `EPOCHS` | `300` | Numero de epocas |
| `LR` | `0.5` | Tasa de aprendizaje |
| `L2` | `0.001` | Regularizacion L2 |
| `DATA_PATH` | `/data/crimes_clean.csv` | Ruta al CSV limpio |
| `MODEL_OUT` | `/out/crime_model_distributed.json` | Ruta de salida del modelo |

El modelo resultante se guarda en `crime_model_distributed.json`.

### Cluster de inferencia (Parte C)

Cada nodo de inferencia (`cmd/inference`) carga el modelo y el scaler, atiende predicciones por TCP y se autorregistra en Redis con TTL. Si el nodo muere, su clave expira y el balanceador deja de verlo automaticamente.

El balanceador (`internal/balancer`) vive embebido en la API: consulta Redis para obtener los nodos vivos, desempata con shuffle y elige el de menor carga efectiva.

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
| `-epochs` | `300` | Numero de epocas |
| `-lr` | `0.5` | Tasa de aprendizaje |
| `-l2` | `0.001` | Regularizacion L2 (lambda) |
| `-test` | `0.2` | Fraccion para test |
| `-threshold` | `0.5` | Umbral de decision |
| `-workers` | `numCPU` | Goroutines para el gradiente |
| `-seed` | `42` | Semilla para reproducibilidad |
| `-load` | `false` | Cargar modelo guardado y evaluar sin reentrenar |

Para evaluar el modelo serializado sin reentrenar:

```bash
go run cmd/train_crimes/main.go -load
```

### Pruebas de usuario (curl)

```bash
# Liveness probe
curl http://localhost:8080/health

# Estado del cluster
curl http://localhost:8080/metrics

# Prediccion — zona de alta peligrosidad
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"district":3,"grid_lat":41.85,"grid_lon":-87.65,"hour":22,"weekday":5,"month":6,"is_night":1,"location_encoded":0,"domestic":0}}'

# Prediccion — zona de baja peligrosidad
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"district":10,"grid_lat":41.90,"grid_lon":-87.70,"hour":9,"weekday":2,"month":3,"is_night":0,"location_encoded":1,"domestic":1}}'
```

---

## Guia de contribucion

Ver [CONTRIBUTING.md](CONTRIBUTING.md).
