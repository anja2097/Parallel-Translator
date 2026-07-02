#include "hwy/highway.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

namespace hn = hwy::HWY_NAMESPACE;

void transpose(const std::vector<double>& A,
               std::vector<double>& B,
               int M, int N)
{
    const hn::ScalableTag<double> d;
    const size_t vlen = hn::Lanes(d);
    const size_t BLOCK_SIZE = 256;

    std::vector<double> temp(BLOCK_SIZE * BLOCK_SIZE);

    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            int i1 = static_cast<int>(std::min(static_cast<size_t>(i0) + BLOCK_SIZE, static_cast<size_t>(M)));
            int j1 = static_cast<int>(std::min(static_cast<size_t>(j0) + BLOCK_SIZE, static_cast<size_t>(N)));
            int block_rows = i1 - i0;
            int block_cols = j1 - j0;

            for (int j = j0; j < j1; ++j) {
                size_t i = i0;
                for (; i + vlen <= static_cast<size_t>(i1); i += vlen) {
                    auto vec = hn::LoadU(d, A.data() + i * N + j);
                    hn::StoreU(vec, d, temp.data() + (j - j0) * block_rows + (i - i0));
                }
                if (i < static_cast<size_t>(i1)) {
                    size_t rem = static_cast<size_t>(i1) - i;
                    auto mask = hn::FirstN(d, rem);
                    auto vec = hn::MaskedLoad(mask, d, A.data() + i * N + j);
                    hn::BlendedStore(vec, mask, d, temp.data() + (j - j0) * block_rows + (i - i0));
                }
            }

            for (int j = j0; j < j1; ++j) {
                int j_block = j - j0;
                double* src = &temp[j_block * block_rows];
                double* dst = &B[j * M + i0];
                size_t remaining = static_cast<size_t>(block_rows);
                size_t i = 0;
                for (; i + vlen <= remaining; i += vlen) {
                    auto vec = hn::LoadU(d, src + i);
                    hn::StoreU(vec, d, dst + i);
                }
                if (i < remaining) {
                    size_t rem = remaining - i;
                    auto mask = hn::FirstN(d, rem);
                    auto vec = hn::MaskedLoad(mask, d, src + i);
                    hn::BlendedStore(vec, mask, d, dst + i);
                }
            }
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

    return 0;
}