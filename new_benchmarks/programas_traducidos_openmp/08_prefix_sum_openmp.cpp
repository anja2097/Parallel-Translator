#include<iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

// Prefix sum (scan inclusivo): out[i] = sum(in[0..i])
void prefix_sum(const std::vector<double>& in,
                std::vector<double>& out,
                int N)
{
    out[0] = in[0];
    for (int i = 1; i < N; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

void init_random(std::vector<double>& data, int n)
{
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double val;
        #pragma omp critical
        val = static_cast<double>(rand()) / RAND_MAX;
        data[i] = val;
    }
}

int main()
{
    const int N = 100000000;  // 100M elementos

    std::vector<double> input(N);
    std::vector<double> output(N, 0.0);

    srand(42);
    init_random(input, N);

    auto start = std::chrono::high_resolution_clock::now();

    prefix_sum(input, output, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Prefix Sum: "
              << elapsed.count() << " ms\n";

    return 0;
}