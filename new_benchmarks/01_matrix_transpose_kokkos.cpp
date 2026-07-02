#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <cstdlib>

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc, argv);

    const int M = 4096;
    const int N = 4096;

    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // Device views
    Kokkos::View<double*, memory_space> d_A("A", M * N);
    Kokkos::View<double*, memory_space> d_B("B", N * M);

    // Host mirrors for initialization
    auto h_A = Kokkos::create_mirror_view(d_A);
    auto h_B = Kokkos::create_mirror_view(d_B);

    // Initialize A with random numbers on host
    std::srand(42);
    for (int i = 0; i < M * N; ++i) {
        h_A(i) = static_cast<double>(std::rand()) / RAND_MAX;
    }

    // Copy A to device, B is already allocated (initialized to 0)
    Kokkos::deep_copy(d_A, h_A);
    Kokkos::deep_copy(d_B, h_B); // ensures B is zeroed on device

    // Transpose: B[j*M + i] = A[i*N + j]
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "transpose",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>( {0,0}, {M,N} ),
        KOKKOS_LAMBDA (int i, int j) {
            d_B(j * M + i) = d_A(i * N + j);
        });
    exec_space().fence();
    double elapsed_ms = timer.seconds() * 1000.0;

    // Compute checksum of B
    double checksum = 0.0;
    Kokkos::parallel_reduce(
        "checksum",
        Kokkos::RangePolicy<>(0, N * M),
        KOKKOS_LAMBDA (int idx, double& local_sum) {
            local_sum += d_B(idx);
        }, checksum);
    exec_space().fence();

    // Copy checksum result back to host (already on host)

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Tiempo de ejecucion de la Transpose: "
              << elapsed_ms << " ms\n";
    std::cout << std::fixed << std::setprecision(6)
              << "Checksum: " << checksum << "\n";

    return 0;
}