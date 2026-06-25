# Build context = raíz del repo (ver docker-compose.distributed.yml), porque el
# binario importa internal/*. Imagen final mínima con solo el binario.
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /bin/coordinator ./cmd/coordinator

FROM alpine:latest
COPY --from=build /bin/coordinator /bin/coordinator
ENTRYPOINT ["/bin/coordinator"]
