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
    if (!mat.data) {
        fprintf(stderr, "Memory allocation failed for rows.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        mat.data[i] = (double*)malloc(n * sizeof(double));
        if (!mat.data[i]) {
            fprintf(stderr, "Memory allocation failed for row %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }
    return mat;
}


void destroy_matrix(Matrix mat) {
    if (mat.data) {
        for (int i = 0; i < mat.size; ++i) {
            free(mat.data[i]);
        }
        free(mat.data);
    }
}


void random_fill(Matrix mat) {
    for (int i = 0; i < mat.size; ++i) {
        for (int j = 0; j < mat.size; ++j) {
            mat.data[i][j] = rand() % 100;
        }
    }
}


void write_to_file(const char* prefix, Matrix mat, int exp_index) {
    char filename[64];
    sprintf(filename, "%s_%d_%d.txt", prefix, exp_index, mat.size);
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open file %s for writing.\n", filename);
        return;
    }
    fprintf(f, "%d\n", mat.size);
    for (int i = 0; i < mat.size; ++i) {
        for (int j = 0; j < mat.size; ++j) {
            fprintf(f, "%.0f ", mat.data[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}


void parallel_multiply(Matrix A, Matrix B, Matrix C) {
    int n = A.size;
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A.data[i][k] * B.data[k][j];
            }
            C.data[i][j] = sum;
        }
    }
}


void save_statistics(const char* filename, int* dims, double* times, int count) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing.\n", filename);
        return;
    }

    fprintf(f, "Parallel Matrix Multiplication Benchmark\n");
    fprintf(f, "========================================\n\n");
    fprintf(f, "| Dimension | Time (seconds) |\n");
    fprintf(f, "|-----------|----------------|\n");
    for (int i = 0; i < count; ++i) {
        fprintf(f, "| %-9d | %14.6f |\n", dims[i], times[i]);
    }
    fprintf(f, "\n\nCSV format:\n");
    fprintf(f, "Dimension,Time\n");
    for (int i = 0; i < count; ++i) {
        fprintf(f, "%d,%.6f\n", dims[i], times[i]);
    }
    fclose(f);
}

int main() {
    int dimensions[] = { 200, 400, 800, 1200, 1600, 2000 };
    int num_experiments = sizeof(dimensions) / sizeof(dimensions[0]);

    double* timings = (double*)malloc(num_experiments * sizeof(double));
    if (!timings) {
        fprintf(stderr, "Memory allocation failed for timings.\n");
        return EXIT_FAILURE;
    }

    printf("========================================\n");
    printf("Parallel Matrix Multiplication Benchmark\n");
    printf("========================================\n\n");

    int max_threads = omp_get_max_threads();
    printf("OpenMP threads available: %d\n", max_threads);
    printf("----------------------------------------\n");

    srand((unsigned)time(NULL));

    for (int exp = 0; exp < num_experiments; ++exp) {
        int N = dimensions[exp];
        printf("\n--- Experiment %d: Matrix size %d x %d ---\n", exp + 1, N, N);

        Matrix matA = create_matrix(N);
        Matrix matB = create_matrix(N);
        Matrix matC = create_matrix(N);

        printf("Generating random matrices...\n");
        random_fill(matA);
        random_fill(matB);

        write_to_file("matrixA", matA, exp + 1);
        write_to_file("matrixB", matB, exp + 1);
        printf("Input matrices saved.\n");

        printf("Starting parallel multiplication...\n");
        double start_time = omp_get_wtime();
        parallel_multiply(matA, matB, matC);
        double end_time = omp_get_wtime();

        double elapsed = end_time - start_time;
        timings[exp] = elapsed;

        printf("Time: %.6f seconds\n", elapsed);

        write_to_file("matrixC", matC, exp + 1);
        printf("Result matrix saved.\n");

        destroy_matrix(matA);
        destroy_matrix(matB);
        destroy_matrix(matC);

        printf("----------------------------------------\n");
    }

    save_statistics("benchmark_results.txt", dimensions, timings, num_experiments);
    printf("\nResults saved to 'benchmark_results.txt'.\n");

    printf("\nFINAL RESULTS:\n");
    printf("+-------------+------------------+\n");
    printf("| Dimension   | Time (seconds)   |\n");
    printf("+-------------+------------------+\n");
    for (int i = 0; i < num_experiments; ++i) {
        printf("| %-11d | %16.6f |\n", dimensions[i], timings[i]);
    }
    printf("+-------------+------------------+\n");

    free(timings);
    printf("\nBenchmark finished.\n");
    return 0;
}