#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

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
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<double>(rand()) / RAND_MAX;
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

    // Suma de control del resultado
    // Muestreo cada 1000 elementos para evitar overflow en la suma total
    double checksum = 0.0;
    for (int i = 0; i < N; i += 1000) checksum += output[i];
    std::cout << std::fixed << std::setprecision(6)
              << "Checksum: " << checksum << "\n";

    return 0;
}
