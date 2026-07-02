#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

// Calcula el histograma de un array de valores enteros
void histogram(const std::vector<int>& data,
               std::vector<int>& hist,
               int num_bins)
{
    for (int i = 0; i < static_cast<int>(hist.size()); ++i) {
        hist[i] = 0;
    }

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        int bin = data[i] % num_bins;
        #pragma omp atomic
        hist[bin]++;
    }
}

void init_random(std::vector<int>& data, int n, int max_val)
{
    for (int i = 0; i < n; ++i) {
        data[i] = rand() % max_val;
    }
}

int main()
{
    const int N = 100000000;  // 100M elementos
    const int NUM_BINS = 256;

    std::vector<int> data(N);
    std::vector<int> hist(NUM_BINS, 0);

    srand(42);
    init_random(data, N, NUM_BINS * 100);

    auto start = std::chrono::high_resolution_clock::now();

    histogram(data, hist, NUM_BINS);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Histogram: "
              << elapsed.count() << " ms\n";

    return 0;
}