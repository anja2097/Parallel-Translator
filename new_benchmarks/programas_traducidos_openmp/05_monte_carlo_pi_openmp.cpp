#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

// Estimación de Pi por Monte Carlo
// Genera puntos aleatorios en [0,1)x[0,1) y cuenta los que caen dentro del
// cuarto de círculo unitario
double monte_carlo_pi(int num_samples)
{
    int inside = 0;

    #pragma omp parallel for
    for (int i = 0; i < num_samples; ++i) {
        double x, y;
        #pragma omp critical
        {
            x = static_cast<double>(rand()) / RAND_MAX;
            y = static_cast<double>(rand()) / RAND_MAX;
        }
        if (x * x + y * y <= 1.0) {
            #pragma omp atomic
            inside++;
        }
    }

    return 4.0 * static_cast<double>(inside) / num_samples;
}

int main()
{
    const int NUM_SAMPLES = 100000000;  // 100M muestras

    srand(42);

    auto start = std::chrono::high_resolution_clock::now();

    double pi_est = monte_carlo_pi(NUM_SAMPLES);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Monte Carlo Pi: "
              << elapsed.count() << " ms\n";
    std::cout << "Pi estimado: " << std::setprecision(6) << pi_est << "\n";

    return 0;
}