#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

// SpMV en formato CSR (Compressed Sparse Row): y = A * x
void spmv_csr(const std::vector<double>& values,
              const std::vector<int>& col_idx,
              const std::vector<int>& row_ptr,
              const std::vector<double>& x,
              std::vector<double>& y,
              int num_rows)
{
    #pragma omp parallel for
    for (int i = 0; i < num_rows; ++i) {
        double sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

// Genera una matriz dispersa aleatoria en formato CSR
void generate_sparse_matrix(std::vector<double>& values,
                            std::vector<int>& col_idx,
                            std::vector<int>& row_ptr,
                            int N, int nnz_per_row)
{
    row_ptr.resize(N + 1);
    row_ptr[0] = 0;

    for (int i = 0; i < N; ++i) {
        int nnz = nnz_per_row;
        for (int j = 0; j < nnz; ++j) {
            int col = rand() % N;
            double val = static_cast<double>(rand()) / RAND_MAX;
            values.push_back(val);
            col_idx.push_back(col);
        }
        row_ptr[i + 1] = row_ptr[i] + nnz;
    }
}

int main()
{
    const int N = 100000;       // filas/columnas
    const int NNZ_PER_ROW = 50; // elementos no-cero por fila

    std::vector<double> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    std::vector<double> x(N);
    std::vector<double> y(N, 0.0);

    srand(42);
    generate_sparse_matrix(values, col_idx, row_ptr, N, NNZ_PER_ROW);
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Repetir para tener tiempo medible
    for (int iter = 0; iter < 100; ++iter) {
        spmv_csr(values, col_idx, row_ptr, x, y, N);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del SpMV CSR (100 iters): "
              << elapsed.count() << " ms\n";

    return 0;
}