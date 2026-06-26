# Build context = raíz del repo (ver docker-compose). Imagen final mínima.
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /bin/inference ./cmd/inference

FROM alpine:latest
COPY --from=build /bin/inference /bin/inference
ENTRYPOINT ["/bin/inference"]
