#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    const int NUM_THREADS = 5;
    omp_set_num_threads(NUM_THREADS);

    if (argc < 2) { 
        printf("Использование: %s N\n", argv[0]); return 1; 
    }

    int N = atoi(argv[1]);
    if (N <= 0) { 
        printf("N должен быть > 0\n"); return 1; 
    }


    size_t NN = (size_t)N * (size_t)N;
    double *A = (double*)malloc(NN * sizeof(double));
    double *B = (double*)malloc(NN * sizeof(double));
    double *C = (double*)malloc(NN * sizeof(double));
    if (!A || !B || !C) { 
        printf("Не хватает памяти\n"); return 1; 
    }

    int i, j, k;

    #pragma omp parallel for private(j) schedule(static)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i*N + j] = i + 1;
            B[i*N + j] = 1.0 / (j + 1);
            C[i*N + j] = 0.0;
        }

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    #pragma omp parallel for private(j,k) schedule(static)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            double s = 0.0;
            for (k = 0; k < N; k++)
                s += A[i*N + k] * B[k*N + j];
            C[i*N + j] = s;
        }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double sec = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("n = %d\n", N);
    printf("C[0][0] = %.6f\n", C[0]);
    printf("C[0][%d] = %.6f\n", N-1, C[(N-1)]);
    printf("C[%d][0] = %.6f\n", N-1, C[(N-1)*N + 0]);
    printf("C[%d][%d] = %.6f\n", N-1, N-1, C[(N-1)*N + (N-1)]);
    printf("Потоков: %d\n", NUM_THREADS);
    printf("Время: %.6f сек\n", sec);
    printf("Производительность: %.2f GFlops\n", (2.0 * N * N * N) / (sec * 1e9));

    free(A); free(B); free(C);
    return 0;
}
