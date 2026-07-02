#include<iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <omp.h>

// K-Nearest Neighbors por fuerza bruta
// Para cada query, calcula la distancia euclidiana a todos los puntos de datos
// y devuelve los índices de los K más cercanos
void knn_bruteforce(const std::vector<double>& data,
                    const std::vector<double>& queries,
                    std::vector<int>& indices,
                    int N, int Q, int D, int K_nn)
{
    std::vector<std::pair<double, int>> dists(N);

    #pragma omp parallel for private(dists)
    for (int q = 0; q < Q; ++q) {
        for (int i = 0; i < N; ++i) {
            double dist = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = queries[q * D + d] - data[i * D + d];
                dist += diff * diff;
            }
            dists[i] = {dist, i};
        }

        std::partial_sort(dists.begin(), dists.begin() + K_nn, dists.end());

        for (int k = 0; k < K_nn; ++k) {
            indices[q * K_nn + k] = dists[k].second;
        }
    }
}

void init_random(std::vector<double>& mat, int n)
{
    for (int i = 0; i < n; ++i) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int N = 20000;    // puntos de datos
    const int Q = 500;      // queries
    const int D = 64;       // dimensiones
    const int K_nn = 10;    // vecinos

    std::vector<double> data(N * D);
    std::vector<double> queries(Q * D);
    std::vector<int> indices(Q * K_nn, 0);

    srand(42);
    init_random(data, N * D);
    init_random(queries, Q * D);

    auto start = std::chrono::high_resolution_clock::now();

    knn_bruteforce(data, queries, indices, N, Q, D, K_nn);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del KNN Brute Force: "
              << elapsed.count() << " ms\n";

    return 0;
}