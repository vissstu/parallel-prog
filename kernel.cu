#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

typedef struct {
    double* data;
    int size;
} Matrix;

Matrix create_matrix(int n) {
    Matrix m;
    m.size = n;
    m.data = (double*)malloc(n * n * sizeof(double));
    if (!m.data) { fprintf(stderr, "Memory allocation failed\n"); exit(1); }
    return m;
}

void destroy_matrix(Matrix m) {
    if (m.data) free(m.data);
}

void random_fill(Matrix m) {
    for (int i = 0; i < m.size * m.size; ++i)
        m.data[i] = rand() % 10;
}

void write_to_file(const char* name, Matrix m) {
    FILE* f = fopen(name, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", name); return; }
    fprintf(f, "%d\n", m.size);
    for (int i = 0; i < m.size; ++i) {
        for (int j = 0; j < m.size; ++j)
            fprintf(f, "%.0f ", m.data[i * m.size + j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

__global__ void mul_kernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    srand((unsigned)time(NULL));

    int dims[] = { 1024, 2048, 3072 };
    int blocks[] = { 8, 16, 32 };
    int ndims = sizeof(dims) / sizeof(dims[0]);
    int nblocks = sizeof(blocks) / sizeof(blocks[0]);
    double results[3][3];

    printf("=====================================================\n");
    printf("  CUDA Matrix Multiplication – Benchmark (C style)\n");
    printf("=====================================================\n\n");

    int devcnt;
    cudaGetDeviceCount(&devcnt);
    if (devcnt == 0) { fprintf(stderr, "No CUDA device\n"); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.0f MB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0));

    FILE* csv = fopen("cuda_benchmark_auto.csv", "w");
    if (csv) fprintf(csv, "Size,BlockSize,Time_sec,GFLOPS\n");

    for (int i = 0; i < ndims; ++i) {
        int n = dims[i];
        printf("=== Size: %dx%d ===\n", n, n);

        Matrix hA = create_matrix(n);
        Matrix hB = create_matrix(n);
        Matrix hC = create_matrix(n);
        random_fill(hA);
        random_fill(hB);

        char nameA[32], nameB[32];
        sprintf(nameA, "matrix_a_%d.txt", n);
        sprintf(nameB, "matrix_b_%d.txt", n);
        write_to_file(nameA, hA);
        write_to_file(nameB, hB);
        printf("Generated: %s, %s\n", nameA, nameB);

        double* dA, * dB, * dC;
        size_t sz = n * n * sizeof(double);
        cudaMalloc(&dA, sz);
        cudaMalloc(&dB, sz);
        cudaMalloc(&dC, sz);
        cudaMemcpy(dA, hA.data, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data, sz, cudaMemcpyHostToDevice);

        for (int j = 0; j < nblocks; ++j) {
            int bs = blocks[j];
            dim3 thr(bs, bs);
            dim3 blk((n + bs - 1) / bs, (n + bs - 1) / bs);

            mul_kernel << <blk, thr >> > (dA, dB, dC, n);
            cudaDeviceSynchronize();

            const int reps = 3;
            double total = 0.0;
            for (int r = 0; r < reps; ++r) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                mul_kernel << <blk, thr >> > (dA, dB, dC, n);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total += ms / 1000.0;
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            double avg = total / reps;
            results[i][j] = avg;
            long long ops = 2LL * n * n * n;
            double gflops = ops / (avg * 1e9);
            printf("  Block %dx%d : %.4f sec, %.2f GFLOPS\n", bs, bs, avg, gflops);

            if (csv) fprintf(csv, "%d,%d,%.6f,%.2f\n", n, bs, avg, gflops);
        }

        cudaMemcpy(hC.data, dC, sz, cudaMemcpyDeviceToHost);
        char nameC[32];
        sprintf(nameC, "matrix_c_%d.txt", n);
        write_to_file(nameC, hC);
        printf("    Result saved: %s\n", nameC);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        destroy_matrix(hA); destroy_matrix(hB); destroy_matrix(hC);
        printf("----------------------------------------\n");
    }

    if (csv) fclose(csv);

    printf("\n==================== FINAL TABLE ====================\n");
    printf("Size\\Block |");
    for (int j = 0; j < nblocks; ++j) printf(" %8dx%d |", blocks[j], blocks[j]);
    printf("\n------------------------------------------------------\n");
    for (int i = 0; i < ndims; ++i) {
        printf("%8d |", dims[i]);
        for (int j = 0; j < nblocks; ++j) {
            printf(" %10.4f |", results[i][j]);
        }
        printf("\n");
    }
    printf("======================================================\n");
    printf("\nResults also saved to cuda_benchmark_auto.csv\n");

    return 0;
}