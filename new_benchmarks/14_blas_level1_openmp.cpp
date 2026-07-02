#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>

// SAXPY: Y = a * X + Y (operación BLAS nivel 1)
void saxpy(double a,
           const std::vector<double>& X,
           std::vector<double>& Y,
           int N)
{
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Y[i] = a * X[i] + Y[i];
    }
}

// Dot product: result = X . Y
double dot_product(const std::vector<double>& X,
                   const std::vector<double>& Y,
                   int N)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += X[i] * Y[i];
    }
    return sum;
}

// Norma L2: ||X||_2
double norm_l2(const std::vector<double>& X, int N)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += X[i] * X[i];
    }
    return std::sqrt(sum);
}

void init_random(std::vector<double>& data, int n)
{
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

int main()
{
    const int N = 100000000;   // 100M elementos
    const int REPS = 10;       // repeticiones para acumular tiempo

    std::vector<double> X(N), Y(N);

    srand(42);
    init_random(X, N);
    init_random(Y, N);

    double a = 2.5;

    auto start = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < REPS; ++r) {
        saxpy(a, X, Y, N);
        double d = dot_product(X, Y, N);
        double n = norm_l2(Y, N);
        // Evitar que el compilador optimice
        if (d == -999.999 && n == -999.999) std::cout << "unreachable";
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del BLAS Level-1 (SAXPY+dot+norm, "
              << REPS << " reps): " << elapsed.count() << " ms\n";

    return 0;
}