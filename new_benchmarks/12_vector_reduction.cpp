#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <limits>

// Reducción: calcula suma, mínimo, máximo y media de un vector
struct ReductionResult {
    double sum;
    double min_val;
    double max_val;
    double mean;
    double variance;
};

ReductionResult vector_reduction(const std::vector<double>& data, int N)
{
    double sum = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    // Primera pasada: sum, min, max
    for (int i = 0; i < N; ++i) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    double mean = sum / N;

    // Segunda pasada: varianza
    double var_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = data[i] - mean;
        var_sum += diff * diff;
    }

    return {sum, min_val, max_val, mean, var_sum / N};
}

void init_random(std::vector<double>& data, int n)
{
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<double>(rand()) / RAND_MAX * 1000.0 - 500.0;
    }
}

int main()
{
    const int N = 200000000;  // 200M elementos

    std::vector<double> data(N);

    srand(42);
    init_random(data, N);

    auto start = std::chrono::high_resolution_clock::now();

    ReductionResult result = vector_reduction(data, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion de la Reduction: "
              << elapsed.count() << " ms\n";

    // Suma de control del resultado
    double checksum = result.sum + result.min_val + result.max_val
                    + result.mean + result.variance;
    std::cout << std::fixed << std::setprecision(6)
              << "Checksum: " << checksum << "\n";

    return 0;
}
