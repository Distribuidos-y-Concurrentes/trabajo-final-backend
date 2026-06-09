package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type Record struct {
	id         int
	crimeType  string
	locType    int
	arrest     bool
	domestic   bool
	district   int
	lat, lon   float64
	gridLat    float64
	gridLon    float64
	hour       int
	weekday    int
	month      int
	year       int
	isNight    bool
	isHighRisk bool
}

// contadores separados por motivo de descarte
type Contadores struct {
	total       int64
	validos     int64
	sinCoords   int64
	duplicados  int64
	fechaMal    int64
	distritoMal int64
	filaCorta   int64
}

var highRisk = map[string]bool{
	"HOMICIDE": true, "ASSAULT": true, "BATTERY": true,
	"ROBBERY": true, "WEAPONS VIOLATION": true,
	"CRIM SEXUAL ASSAULT": true, "ARSON": true,
}

func parseDate(s string) (time.Time, error) {
	for _, layout := range []string{"01/02/2006 03:04:05 PM", "01/02/2006 15:04:05"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("fecha invalida")
}

func encodeLocation(s string) int {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch {
	case strings.Contains(s, "STREET") || strings.Contains(s, "SIDEWALK") || strings.Contains(s, "ALLEY"):
		return 0
	case strings.Contains(s, "RESIDENCE") || strings.Contains(s, "APARTMENT"):
		return 1
	case strings.Contains(s, "STORE") || strings.Contains(s, "SHOP") || strings.Contains(s, "MARKET"):
		return 2
	case strings.Contains(s, "RESTAURANT") || strings.Contains(s, "BAR") || strings.Contains(s, "TAVERN"):
		return 3
	case strings.Contains(s, "SCHOOL") || strings.Contains(s, "COLLEGE"):
		return 4
	case strings.Contains(s, "PARKING"):
		return 5
	case strings.Contains(s, "VEHICLE"):
		return 6
	case strings.Contains(s, "CTA"):
		return 7
	default:
		return 8
	}
}

// columnas del dataset real de Chicago (22 columnas):
// 0:ID, 1:Case Number, 2:Date, 3:Block, 4:IUCR, 5:Primary Type,
// 6:Description, 7:Location Description, 8:Arrest, 9:Domestic,
// 10:Beat, 11:District, 12:Ward, 13:Community Area, 14:FBI Code,
// 15:X Coordinate, 16:Y Coordinate, 17:Year, 18:Updated On,
// 19:Latitude, 20:Longitude, 21:Location

func procesarChunk(path string, offset, end int64, out chan<- Record, vistos *sync.Map,
	c *Contadores, wg *sync.WaitGroup) {
	defer wg.Done()

	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	f.Seek(offset, 0)
	reader := csv.NewReader(io.LimitReader(f, end-offset))
	reader.LazyQuotes = true
	reader.FieldsPerRecord = -1
	reader.ReuseRecord = true

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		atomic.AddInt64(&c.total, 1)

		if err != nil || len(row) < 22 {
			atomic.AddInt64(&c.filaCorta, 1)
			continue
		}

		id, err := strconv.Atoi(strings.TrimSpace(row[0]))
		if err != nil || id <= 0 {
			atomic.AddInt64(&c.filaCorta, 1)
			continue
		}

		if _, existe := vistos.LoadOrStore(id, true); existe {
			atomic.AddInt64(&c.duplicados, 1)
			continue
		}

		lat, err1 := strconv.ParseFloat(strings.TrimSpace(row[19]), 64)
		lon, err2 := strconv.ParseFloat(strings.TrimSpace(row[20]), 64)
		if err1 != nil || err2 != nil || lat < 41.6 || lat > 42.1 || lon < -88.0 || lon > -87.4 {
			atomic.AddInt64(&c.sinCoords, 1)
			continue
		}

		dt, err := parseDate(strings.TrimSpace(row[2]))
		if err != nil {
			atomic.AddInt64(&c.fechaMal, 1)
			continue
		}

		distrito, err := strconv.Atoi(strings.TrimSpace(row[11]))
		if err != nil || distrito < 1 || distrito > 31 {
			atomic.AddInt64(&c.distritoMal, 1)
			continue
		}

		crimeType := strings.ToUpper(strings.TrimSpace(row[5]))
		hour := dt.Hour()

		rec := Record{
			id:         id,
			crimeType:  crimeType,
			locType:    encodeLocation(row[7]),
			arrest:     strings.ToUpper(strings.TrimSpace(row[8])) == "TRUE",
			domestic:   strings.ToUpper(strings.TrimSpace(row[9])) == "TRUE",
			district:   distrito,
			lat:        lat,
			lon:        lon,
			gridLat:    math.Round(lat/0.004) * 0.004,
			gridLon:    math.Round(lon/0.004) * 0.004,
			hour:       hour,
			weekday:    int(dt.Weekday()),
			month:      int(dt.Month()),
			year:       dt.Year(),
			isNight:    hour >= 20 || hour < 6,
			isHighRisk: highRisk[crimeType],
		}

		out <- rec
		atomic.AddInt64(&c.validos, 1)
	}
}

func main() {
	inputFile  := "crimes.csv"
	outputFile := "crimes_clean.csv"
	numWorkers := 4

	fmt.Println("iniciando limpieza...")

	f, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("no se encontro crimes.csv")
		os.Exit(1)
	}

	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.Read()
	headerEnd, _ := f.Seek(0, io.SeekCurrent)
	fi, _ := f.Stat()
	totalSize := fi.Size()
	f.Close()

	chunkSize := (totalSize - headerEnd) / int64(numWorkers)
	offsets := make([]int64, numWorkers+1)
	offsets[0] = headerEnd
	offsets[numWorkers] = totalSize

	buf := make([]byte, 256)
	faux, _ := os.Open(inputFile)
	for i := 1; i < numWorkers; i++ {
		pos := headerEnd + int64(i)*chunkSize
		faux.Seek(pos, 0)
		n, _ := faux.Read(buf)
		for j := 0; j < n; j++ {
			pos++
			if buf[j] == '\n' {
				break
			}
		}
		offsets[i] = pos
	}
	faux.Close()

	out := make(chan Record, 20000)
	var c Contadores
	var vistos sync.Map
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go procesarChunk(inputFile, offsets[i], offsets[i+1], out, &vistos, &c, &wg)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	outFile, _ := os.Create(outputFile)
	writer := csv.NewWriter(outFile)
	writer.Write([]string{
		"id", "crime_type", "location_encoded", "arrest", "domestic",
		"district", "latitude", "longitude", "grid_lat", "grid_lon",
		"hour", "weekday", "month", "year", "is_night", "is_high_risk",
	})

	inicio := time.Now()
	var escritos int64

	for rec := range out {
		writer.Write([]string{
			strconv.Itoa(rec.id),
			rec.crimeType,
			strconv.Itoa(rec.locType),
			strconv.FormatBool(rec.arrest),
			strconv.FormatBool(rec.domestic),
			strconv.Itoa(rec.district),
			strconv.FormatFloat(rec.lat, 'f', 7, 64),
			strconv.FormatFloat(rec.lon, 'f', 7, 64),
			strconv.FormatFloat(rec.gridLat, 'f', 6, 64),
			strconv.FormatFloat(rec.gridLon, 'f', 6, 64),
			strconv.Itoa(rec.hour),
			strconv.Itoa(rec.weekday),
			strconv.Itoa(rec.month),
			strconv.Itoa(rec.year),
			strconv.FormatBool(rec.isNight),
			strconv.FormatBool(rec.isHighRisk),
		})
		escritos++
	}

	writer.Flush()
	outFile.Close()

	descartados := c.total - c.validos
	pct := float64(c.validos) / float64(c.total) * 100

	fmt.Printf("\n--- reporte de limpieza ---\n")
	fmt.Printf("tiempo:               %s\n", time.Since(inicio).Round(time.Millisecond))
	fmt.Printf("registros leidos:     %d\n", c.total)
	fmt.Printf("registros validos:    %d (%.1f%%)\n", c.validos, pct)
	fmt.Printf("registros descartados:%d\n\n", descartados)
	fmt.Printf("motivo de descarte:\n")
	fmt.Printf("  sin coordenadas:    %d\n", c.sinCoords)
	fmt.Printf("  duplicados:         %d\n", c.duplicados)
	fmt.Printf("  fecha invalida:     %d\n", c.fechaMal)
	fmt.Printf("  distrito invalido:  %d\n", c.distritoMal)
	fmt.Printf("  fila incompleta:    %d\n\n", c.filaCorta)
	fmt.Printf("archivo guardado: %s\n", outputFile)
}