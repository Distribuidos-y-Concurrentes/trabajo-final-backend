# Guia de contribucion

## Flujo de trabajo

El repositorio usa Git Flow con dos ramas permanentes:

- `main` — codigo estable; solo recibe merges desde `develop` mediante PR aprobado
- `develop` — rama de integracion; aqui se integran los features antes de pasar a produccion

Ramas de trabajo:

| Prefijo | Uso |
|---------|-----|
| `feature/<descripcion>` | Nueva funcionalidad |
| `fix/<descripcion>` | Correccion de bug |
| `docs/<descripcion>` | Cambios de documentacion |
| `chore/<descripcion>` | Configuracion, dependencias, refactoring |

Crear una rama de trabajo:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/mi-feature
```

## Commits

Formato: `<tipo>: <descripcion en imperativo, minusculas>`

Tipos validos:

| Tipo | Cuando usarlo |
|------|---------------|
| `feat` | Nueva funcionalidad |
| `fix` | Correccion de bug |
| `docs` | Solo documentacion |
| `refactor` | Reestructuracion sin cambio de comportamiento |
| `test` | Agregar o corregir tests |
| `chore` | Configuracion, dependencias |

Ejemplos:

```
feat: add concurrent chunk reader for large CSVs
fix: avoid race condition in StandardScaler Fit
docs: document logreg functional options API
test: add unit tests for metrics package
```

Un commit debe contener un cambio atomico. No mezcles funcionalidades distintas en un solo commit.

## Pull Requests

1. Abre el PR hacia `develop`, no hacia `main`
2. El titulo sigue el mismo formato que los commits
3. La descripcion debe explicar el que y el por que del cambio
4. El PR debe pasar `go vet ./...` y `go test ./...` antes de ser revisado
5. Requiere al menos una aprobacion del equipo para hacer merge

## Estandares de codigo

- Formatea con `gofmt` antes de commitear
- Corre `go vet ./...` para detectar errores estaticos
- Las funciones y tipos exportados llevan un comentario godoc en ingles
- Los comentarios internos pueden estar en espanol
- No commitees archivos de datos ni modelos generados durante el entrenamiento; usa `.gitignore` o Git LFS

## Tests

```bash
go test ./...
```

Cada paquete en `internal/` debe tener su archivo `_test.go`. Los tests no deben depender de archivos de datos externos; usa fixtures pequenas o datos generados en el propio test.

## Reporte de bugs

Abre un issue en GitHub con:

- Descripcion del comportamiento esperado vs el observado
- Pasos minimos para reproducir el problema
- Version de Go (`go version`) y sistema operativo
