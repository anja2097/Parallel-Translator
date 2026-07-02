#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

// Simulación N-body: calcula fuerzas gravitacionales y actualiza posiciones
void nbody_step(std::vector<Body>& bodies, int N, double dt)
{
    const double G = 6.674e-11;
    const double SOFTENING = 1e-9;

    std::vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;

            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;

            double dist_sq = dx * dx + dy * dy + dz * dz + SOFTENING;
            double inv_dist = 1.0 / std::sqrt(dist_sq);
            double inv_dist3 = inv_dist * inv_dist * inv_dist;

            double f = G * bodies[i].mass * bodies[j].mass * inv_dist3;

            fx[i] += f * dx;
            fy[i] += f * dy;
            fz[i] += f * dz;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        bodies[i].vx += dt * fx[i] / bodies[i].mass;
        bodies[i].vy += dt * fy[i] / bodies[i].mass;
        bodies[i].vz += dt * fz[i] / bodies[i].mass;

        bodies[i].x += dt * bodies[i].vx;
        bodies[i].y += dt * bodies[i].vy;
        bodies[i].z += dt * bodies[i].vz;
    }
}

int main()
{
    const int N = 4096;
    const int STEPS = 5;
    const double dt = 0.01;

    std::vector<Body> bodies(N);

    srand(42);
    for (int i = 0; i < N; ++i) {
        bodies[i].x = static_cast<double>(rand()) / RAND_MAX * 100.0;
        bodies[i].y = static_cast<double>(rand()) / RAND_MAX * 100.0;
        bodies[i].z = static_cast<double>(rand()) / RAND_MAX * 100.0;
        bodies[i].vx = 0.0;
        bodies[i].vy = 0.0;
        bodies[i].vz = 0.0;
        bodies[i].mass = static_cast<double>(rand()) / RAND_MAX * 1e6 + 1.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < STEPS; ++s) {
        nbody_step(bodies, N, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del N-Body: "
              << elapsed.count() << " ms\n";

    return 0;
}