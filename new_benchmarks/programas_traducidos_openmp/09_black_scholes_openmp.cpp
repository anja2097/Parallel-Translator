#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// Aproximación de la CDF normal estándar
double norm_cdf(double x)
{
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x >= 0) ? 1 : -1;
    x = std::fabs(x) / std::sqrt(2.0);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Black-Scholes para opciones europeas de compra (call)
void black_scholes(const std::vector<double>& S,    // precio spot
                   const std::vector<double>& K,    // strike
                   const std::vector<double>& T,    // tiempo a expiración
                   const std::vector<double>& r,    // tasa libre de riesgo
                   const std::vector<double>& sigma, // volatilidad
                   std::vector<double>& call_price,
                   int N)
{
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double d1 = (std::log(S[i] / K[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * T[i])
                     / (sigma[i] * std::sqrt(T[i]));
        double d2 = d1 - sigma[i] * std::sqrt(T[i]);

        call_price[i] = S[i] * norm_cdf(d1) - K[i] * std::exp(-r[i] * T[i]) * norm_cdf(d2);
    }
}

int main()
{
    const int N = 10000000;  // 10M opciones

    std::vector<double> S(N), K(N), T(N), r(N), sigma(N);
    std::vector<double> call_price(N, 0.0);

    srand(42);
    for (int i = 0; i < N; ++i) {
        S[i]     = 50.0 + static_cast<double>(rand()) / RAND_MAX * 150.0;
        K[i]     = 50.0 + static_cast<double>(rand()) / RAND_MAX * 150.0;
        T[i]     = 0.1  + static_cast<double>(rand()) / RAND_MAX * 2.0;
        r[i]     = 0.01 + static_cast<double>(rand()) / RAND_MAX * 0.09;
        sigma[i] = 0.1  + static_cast<double>(rand()) / RAND_MAX * 0.5;
    }

    auto start = std::chrono::high_resolution_clock::now();

    black_scholes(S, K, T, r, sigma, call_price, N);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Black-Scholes: "
              << elapsed.count() << " ms\n";

    return 0;
}