#include<iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

// Ecuación del calor 1D por diferencias finitas explícitas
// u_t = alpha * u_xx
// Esquema explícito: u_new[i] = u[i] + r * (u[i-1] - 2*u[i] + u[i+1])
// donde r = alpha * dt / dx^2
void heat_equation_1d(std::vector<double>& u, int N, int steps,
                      double alpha, double dx, double dt)
{
    double r = alpha * dt / (dx * dx);
    std::vector<double> u_new(N, 0.0);

    for (int s = 0; s < steps; ++s) {
        #pragma omp parallel for
        for (int i = 1; i < N - 1; ++i) {
            u_new[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]);
        }
        // Condiciones de contorno fijas
        u_new[0] = u[0];
        u_new[N - 1] = u[N - 1];

        std::swap(u, u_new);
    }
}

int main()
{
    const int N = 100000;
    const int STEPS = 5000;
    const double ALPHA = 0.01;
    const double DX = 1.0 / N;
    const double DT = 0.4 * DX * DX / ALPHA;  // estabilidad CFL

    std::vector<double> u(N, 0.0);

    // Condición inicial: pulso en el centro
    for (int i = N / 4; i < 3 * N / 4; ++i) {
        u[i] = 1.0;
    }
    // Condiciones de contorno
    u[0] = 0.0;
    u[N - 1] = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    heat_equation_1d(u, N, STEPS, ALPHA, DX, DT);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion de la Heat Equation 1D: "
              << elapsed.count() << " ms\n";

    return 0;
}