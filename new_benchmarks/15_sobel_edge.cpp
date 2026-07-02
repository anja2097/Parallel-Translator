#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cmath>

// Convierte imagen RGB a escala de grises
// gray = 0.299*R + 0.587*G + 0.114*B
void rgb_to_gray(const std::vector<double>& rgb,
                 std::vector<double>& gray,
                 int H, int W)
{
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int idx = (i * W + j) * 3;
            gray[i * W + j] = 0.299 * rgb[idx]
                             + 0.587 * rgb[idx + 1]
                             + 0.114 * rgb[idx + 2];
        }
    }
}

// Filtro Sobel para detección de bordes
// Calcula la magnitud del gradiente en cada pixel
void sobel_edge(const std::vector<double>& gray,
                std::vector<double>& edges,
                int H, int W)
{
    // Kernels Sobel
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int i = 1; i < H - 1; ++i) {
        for (int j = 1; j < W - 1; ++j) {
            double sx = 0.0, sy = 0.0;

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    double val = gray[(i + ki) * W + (j + kj)];
                    sx += Gx[ki + 1][kj + 1] * val;
                    sy += Gy[ki + 1][kj + 1] * val;
                }
            }

            edges[i * W + j] = std::sqrt(sx * sx + sy * sy);
        }
    }
}

void init_random_rgb(std::vector<double>& img, int n)
{
    for (int i = 0; i < n; ++i) {
        img[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int H = 4096;
    const int W = 4096;

    std::vector<double> rgb(H * W * 3);
    std::vector<double> gray(H * W, 0.0);
    std::vector<double> edges(H * W, 0.0);

    srand(42);
    init_random_rgb(rgb, H * W * 3);

    auto start = std::chrono::high_resolution_clock::now();

    rgb_to_gray(rgb, gray, H, W);
    sobel_edge(gray, edges, H, W);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion del Sobel Edge Detection: "
              << elapsed.count() << " ms\n";

    // Suma de control del resultado
    double checksum = 0.0;
    for (int i = 0; i < H * W; ++i) checksum += edges[i];
    std::cout << std::fixed << std::setprecision(6)
              << "Checksum: " << checksum << "\n";

    return 0;
}
