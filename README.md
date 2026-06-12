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
| B — Entrenamiento distribuido | Cluster de 3 nodos Go que computan gradientes por TCP | En desarrollo |
| C — API coordinadora | Servidor REST + WebSocket con latencia < 100ms | Pendiente |
| D — Visualizacion | Frontend React con mapas de calor y metricas del cluster | Pendiente |

Infraestructura planificada: MongoDB (persistencia historica), Redis (cache de predicciones).

---

## Estructura del repositorio

```
trabajo-final-backend/
├── cmd/
│   ├── api/              # Punto de entrada de la API coordinadora
│   ├── cleaning/         # Cargador y limpiador concurrente del dataset
│   └── train_crimes/     # CLI de entrenamiento local con metricas
├── internal/
│   ├── dataset/          # TrainTestSplit y StandardScaler
│   ├── logreg/           # Regresion logistica con gradiente paralelo
│   └── metrics/          # Accuracy, Precision, Recall, F1, matriz de confusion
├── crime_model.json      # Pesos del modelo serializado
├── crime_scaler.json     # Parametros del StandardScaler serializado
└── go.mod
```

---

## Entregable 1 — Estado actual

### Limpieza concurrente (Parte A)

El cargador divide el archivo CSV en chunks de bytes y asigna cada chunk a una goroutine. Usa `sync.Map` para deduplicacion por ID sin contention y contadores `atomic` para el reporte final.

Resultados sobre el dataset completo con 4 workers:

| Metrica | Valor |
|---------|-------|
| Registros leidos | 8,567,753 |
| Registros validos | 8,470,795 (98.87%) |
| Tiempo de ejecucion | ~2 min 23 s |

Filtros aplicados: coordenadas fuera del rango de Chicago, distritos fuera del rango 1-31, duplicados por ID, fechas mal formateadas, filas incompletas.

### Modelo de ML (Parte B — local)

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

## Ejecucion

### Requisitos

- Go 1.21 o superior
- `crimes.csv` del Chicago Data Portal ubicado en `cmd/cleaning/`

### Limpieza del dataset

```bash
cd cmd/cleaning
go run limpieza.go
```

Genera `crimes_clean.csv` en el mismo directorio.

### Entrenamiento

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
