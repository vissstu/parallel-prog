#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double* create_matrix(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

void random_fill(double* mat, int n) {
    for (int i = 0; i < n * n; i++)
        mat[i] = rand() % 100;
}

void multiply_block(double* A, double* B, double* C, int n, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    if (rank == 0) {
        printf("MPI Matrix Multiplication\n");
        printf("Processes: %d\n\n", size);
    }

    srand(time(NULL) + rank);

    for (int s = 0; s < num_sizes; s++) {

        int N = sizes[s];

        int rows = N / size;

        double* A = NULL;
        double* B = create_matrix(N, N);
        double* C = NULL;

        double* local_A = create_matrix(rows, N);
        double* local_C = create_matrix(rows, N);

        if (rank == 0) {
            A = create_matrix(N, N);
            C = create_matrix(N, N);

            random_fill(A, N);
            random_fill(B, N);
        }

        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatter(A, rows * N, MPI_DOUBLE,
            local_A, rows * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        multiply_block(local_A, B, local_C, N, rows);

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        MPI_Gather(local_C, rows * N, MPI_DOUBLE,
            C, rows * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Size %d x %d | Time: %.6f sec\n", N, N, end - start);
        }

        free(local_A);
        free(local_C);
        free(B);

        if (rank == 0) {
            free(A);
            free(C);
        }
    }

    MPI_Finalize();
    return 0;
}