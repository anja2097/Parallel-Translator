#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

void transpose(const std::vector<double>& A,
               std::vector<double>& B,
               int M, int N)
{
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

void init_random(std::vector<double>& mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int M = 4096;
    const int N = 4096;

    std::vector<double> A(M * N);
    std::vector<double> B(N * M, 0.0);

    srand(42);
    init_random(A, M, N);

    auto start = std::chrono::high_resolution_clock::now();

    transpose(A, B, M, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion de la Transpose: "
              << elapsed.count() << " ms\n";

    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum)
    for (int i = 0; i < N * M; ++i) checksum += B[i];
    std::cout << std::fixed << std::setprecision(6)
              << "Checksum: " << checksum << "\n";

    return 0;
}