#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

// Calcula el conjunto de Mandelbrot
// Para cada pixel, itera z = z^2 + c y cuenta iteraciones hasta divergencia
void mandelbrot(std::vector<int>& image, int W, int H, int max_iter)
{
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            double c_re = x_min + (x_max - x_min) * col / W;
            double c_im = y_min + (y_max - y_min) * row / H;

            double z_re = 0.0, z_im = 0.0;
            int iter = 0;

            while (z_re * z_re + z_im * z_im <= 4.0 && iter < max_iter) {
                double tmp = z_re * z_re - z_im * z_im + c_re;
                z_im = 2.0 * z_re * z_im + c_im;
                z_re = tmp;
                iter++;
            }

            image[row * W + col] = iter;
        }
    }
}

int main()
{
    const int W = 4096;
    const int H = 4096;
    const int MAX_ITER = 500;

    std::vector<int> image(W * H, 0);

    auto start = std::chrono::high_resolution_clock::now();

    mandelbrot(image, W, H, MAX_ITER);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Mandelbrot: "
              << elapsed.count() << " ms\n";

    return 0;
}