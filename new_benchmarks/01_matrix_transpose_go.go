package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

func main() {
	M := 4096
	N := 4096

	A := make([]float64, M*N)
	B := make([]float64, N*M)

	rand.Seed(42)
	for i := range A {
		A[i] = rand.Float64()
	}

	start := time.Now()

	nWorkers := runtime.NumCPU()
	var wg sync.WaitGroup
	chunkSize := (N + nWorkers - 1) / nWorkers

	for w := 0; w < nWorkers; w++ {
		startJ := w * chunkSize
		endJ := startJ + chunkSize
		if endJ > N {
			endJ = N
		}
		if startJ >= endJ {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				for i := 0; i < M; i++ {
					B[j*M+i] = A[i*N+j]
				}
			}
		}(startJ, endJ)
	}
	wg.Wait()

	elapsed := time.Since(start)

	elapsedMs := float64(elapsed.Nanoseconds()) / 1e6
	fmt.Printf("Tiempo de ejecucion de la Transpose: %.2f ms\n", elapsedMs)

	checksum := 0.0
	for i := 0; i < N*M; i++ {
		checksum += B[i]
	}
	fmt.Printf("Checksum: %.6f\n", checksum)
}