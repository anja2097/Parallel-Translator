#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// Iteración de Jacobi para la ecuación de Laplace 2D
// Promedia los 4 vecinos de cada celda interior
void jacobi_2d(const std::vector<double>& in,
               std::vector<double>& out,
               int N, int iters)
{
    std::vector<double> src(in);
    std::vector<double> dst(N * N, 0.0);

    for (int it = 0; it < iters; ++it) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                dst[i * N + j] = 0.25 * (src[(i - 1) * N + j] +
                                          src[(i + 1) * N + j] +
                                          src[i * N + (j - 1)] +
                                          src[i * N + (j + 1)]);
            }
        }
        std::swap(src, dst);
    }

    out = src;
}

void init_random(std::vector<double>& mat, int n)
{
    for (int i = 0; i < n * n; ++i) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int N = 1024;
    const int ITERS = 100;

    std::vector<double> grid(N * N);
    std::vector<double> result(N * N, 0.0);

    srand(42);
    init_random(grid, N);

    auto start = std::chrono::high_resolution_clock::now();

    jacobi_2d(grid, result, N, ITERS);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Jacobi 2D: "
              << elapsed.count() << " ms\n";

    return 0;
}