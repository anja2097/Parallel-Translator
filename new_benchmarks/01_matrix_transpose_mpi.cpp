#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int M = 4096;
    const int N = 4096;

    int dims[2];
    int dim0 = (int)std::sqrt((double)size);
    while (dim0 > 0 && size % dim0 != 0) --dim0;
    if (dim0 == 0) dim0 = 1;
    dims[0] = dim0;
    dims[1] = size / dim0;

    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int* rows_per_proc = new int[dims[0]];
    int* cols_per_proc = new int[dims[1]];
    int base_rows = M / dims[0];
    int rem_rows = M % dims[0];
    for (int i = 0; i < dims[0]; ++i)
        rows_per_proc[i] = base_rows + (i < rem_rows ? 1 : 0);
    int base_cols = N / dims[1];
    int rem_cols = N % dims[1];
    for (int j = 0; j < dims[1]; ++j)
        cols_per_proc[j] = base_cols + (j < rem_cols ? 1 : 0);

    int my_rows = rows_per_proc[coords[0]];
    int my_cols = cols_per_proc[coords[1]];

    int start_row = 0;
    for (int i = 0; i < coords[0]; ++i) start_row += rows_per_proc[i];
    int start_col = 0;
    for (int j = 0; j < coords[1]; ++j) start_col += cols_per_proc[j];

    long long start_index = (long long)start_row * N + start_col;

    double* sendbuf = new double[my_rows * my_cols];
    double* recvbuf = new double[my_rows * my_cols];

    std::srand(42);
    for (long long k = 0; k < start_index; ++k) std::rand();

    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < my_cols; ++j) {
            double val = double(std::rand()) / RAND_MAX;
            sendbuf[j * my_rows + i] = val;
        }
    }

    int target_coords[2] = {target_coords[0] = coords[1], target_coords[1] = coords[0]};
    int target_rank;
    MPI_Cart_rank(cart_comm, target_coords, &target_rank);

    if (target_rank == rank) {
        for (int i = 0; i < my_rows * my_cols; ++i) recvbuf[i] = sendbuf[i];
    } else {
        int sendcount = my_rows * my_cols;
        MPI_Sendrecv(sendbuf, sendcount, MPI_DOUBLE, target_rank, 0,
                     recvbuf, sendcount, MPI_DOUBLE, target_rank, 0,
                     cart_comm, MPI_STATUS_IGNORE);
    }

    double local_sum = 0.0;
    for (int i = 0; i < my_rows * my_cols; ++i)
        local_sum += recvbuf[i];

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    delete[] sendbuf;
    delete[] recvbuf;
    delete[] rows_per_proc;
    delete[] cols_per_proc;

    MPI_Barrier(MPI_COMM_WORLD);
    double tstart = MPI_Wtime();

    MPI_Cart_coords(cart_comm, rank, 2, coords);
    my_rows = rows_per_proc[coords[0]];
    my_cols = cols_per_proc[coords[1]];
    start_row = 0;
    for (int i = 0; i < coords[0]; ++i) start_row += rows_per_proc[i];
    start_col = 0;
    for (int j = 0; j < coords[1]; ++j) start_col += cols_per_proc[j];
    start_index = (long long)start_row * N + start_col;

    int target_coords[2] = {target_coords[0] = coords[1], target_coords[1] = coords[0]};
    int target_rank;

    sendbuf = new double[my_rows * my_cols];
    recvbuf = new double[my_rows * my_cols];

    std::srand(42);
    for (long long k = 0; k < start_index; ++k) std::rand();

    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < my_cols; ++j) {
            double val = double(std::rand()) / RAND_MAX;
            sendbuf[j * my_rows + i] = val;
        }
    }

    MPI_Cart_rank(cart_comm, target_coords, &target_rank);
    if (target_rank == rank) {
        for (int i = 0; i < my_rows * my_cols; ++i) recvbuf[i] = sendbuf[i];
    } else {
        int sendcount = my_rows * my_cols;
        MPI_Sendrecv(sendbuf, sendcount, MPI_DOUBLE, target_rank, 0,
                     recvbuf, sendcount, MPI_DOUBLE, target_rank, 0,
                     cart_comm, MPI_STATUS_IGNORE);
    }

    local_sum = 0.0;
    for (int i = 0; i < my_rows * my_cols; ++i)
        local_sum += recvbuf[i];

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double tend = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Tiempo de ejecucion de la Transpose: "
                  << (tend - tstart) * 1000.0 << " ms\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "Checksum: " << global_sum << "\n";
    }

    delete[] sendbuf;
    delete[] recvbuf;
    delete[] rows_per_proc;
    delete[] cols_per_proc;
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}