#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef struct {
    double** data;
    int size;
} Matrix;


Matrix create_matrix(int n) {
    Matrix mat;
    mat.size = n;
    mat.data = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; ++i) {
        mat.data[i] = (double*)malloc(n * sizeof(double));
    }
    return mat;
}

void destroy_matrix(Matrix mat) {
    for (int i = 0; i < mat.size; ++i)
        free(mat.data[i]);
    free(mat.data);
}

void random_fill(Matrix mat) {
    for (int i = 0; i < mat.size; ++i)
        for (int j = 0; j < mat.size; ++j)
            mat.data[i][j] = rand() % 100;
}


void write_matrix(const char* prefix, Matrix mat, int exp, int threads) {
    char filename[100];
    sprintf(filename, "%s_%d_%d_t%d.txt", prefix, exp, mat.size, threads);

    FILE* f = fopen(filename, "w");
    fprintf(f, "%d\n", mat.size);

    for (int i = 0; i < mat.size; ++i) {
        for (int j = 0; j < mat.size; ++j)
            fprintf(f, "%.0f ", mat.data[i][j]);
        fprintf(f, "\n");
    }

    fclose(f);
}


void parallel_multiply(Matrix A, Matrix B, Matrix C) {
    int n = A.size;

    // обнуление матрицы C
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C.data[i][j] = 0.0;
        }
    }

    // оптимизированный порядк циклов
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double aik = A.data[i][k];
            for (int j = 0; j < n; ++j) {
                C.data[i][j] += aik * B.data[k][j];
            }
        }
    }
}


int main() {

    srand((unsigned)time(NULL));

    int sizes[] = { 200, 400, 800, 1200, 1600, 2000 };
    int threads_list[] = { 1, 2, 4, 8 };

    int size_count = sizeof(sizes) / sizeof(sizes[0]);
    int thread_count = sizeof(threads_list) / sizeof(threads_list[0]);

    FILE* report = fopen("benchmark_results.csv", "w");
    fprintf(report, "Threads,Size,Time\n");

    printf("=====================================\n");
    // проверка работы OpenMP
    int check_threads = 0;
#pragma omp parallel
    {
#pragma omp single
        check_threads = omp_get_num_threads();
    }
    printf("OpenMP active! Threads in parallel region: %d\n", check_threads);
    if (check_threads == 1) {
        printf("WARNING: OpenMP seems NOT working! Check compiler settings.\n");
    }
    printf("Parallel Matrix Multiplication (OpenMP)\n");
    printf("=====================================\n");

    for (int t = 0; t < thread_count; ++t) {

        int threads = threads_list[t];
        omp_set_num_threads(threads);

        printf("\n===== THREADS: %d =====\n", threads);

        for (int s = 0; s < size_count; ++s) {

            int N = sizes[s];

            printf("Size %d x %d ... ", N, N);

            Matrix A = create_matrix(N);
            Matrix B = create_matrix(N);
            Matrix C = create_matrix(N);

            random_fill(A);
            random_fill(B);

            double start = omp_get_wtime();
            parallel_multiply(A, B, C);
            double end = omp_get_wtime();

            double time = end - start;

            printf("Time: %.6f sec\n", time);

            fprintf(report, "%d,%d,%.6f\n", threads, N, time);

            write_matrix("matrixA", A, s + 1, threads);
            write_matrix("matrixB", B, s + 1, threads);
            write_matrix("matrixC", C, s + 1, threads);

            destroy_matrix(A);
            destroy_matrix(B);
            destroy_matrix(C);
        }
    }

    fclose(report);

    printf("\nResults saved to benchmark_results.csv\n");
    printf("Done.\n");

    return 0;
}
