#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// Convolución 2D: aplica un filtro (kernel) sobre una imagen
// Imagen (H x W), Kernel (KH x KW), Salida (H x W) con zero-padding
void convolution_2d(const std::vector<double>& image,
                    const std::vector<double>& kernel,
                    std::vector<double>& output,
                    int H, int W, int KH, int KW)
{
    int pad_h = KH / 2;
    int pad_w = KW / 2;

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double sum = 0.0;
            for (int ki = 0; ki < KH; ++ki) {
                for (int kj = 0; kj < KW; ++kj) {
                    int ii = i + ki - pad_h;
                    int jj = j + kj - pad_w;

                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                        sum += image[ii * W + jj] * kernel[ki * KW + kj];
                    }
                }
            }
            output[i * W + j] = sum;
        }
    }
}

void init_random(std::vector<double>& mat, int n)
{
    for (int i = 0; i < n; ++i) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int H = 2048;
    const int W = 2048;
    const int KH = 7;
    const int KW = 7;

    std::vector<double> image(H * W);
    std::vector<double> kernel(KH * KW);
    std::vector<double> output(H * W, 0.0);

    srand(42);
    init_random(image, H * W);
    init_random(kernel, KH * KW);

    auto start = std::chrono::high_resolution_clock::now();

    convolution_2d(image, kernel, output, H, W, KH, KW);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion de la Convolution 2D: "
              << elapsed.count() << " ms\n";

    // Suma de control del resultado
    double checksum = 0.0;
    for (int i = 0; i < H * W; ++i) checksum += output[i];
    std::cout << std::fixed << std::setprecision(6)
              << "Checksum: " << checksum << "\n";

    return 0;
}
