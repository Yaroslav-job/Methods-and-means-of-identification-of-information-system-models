#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int N;
    double *A = NULL, *B = NULL, *C = NULL;
    int i, j, k;
    int rank, size;

    if (argc < 2) {
        printf("Использование: %s N\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    if (N <= 0) {
        printf("N должен быть > 0\n");
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Выделяем память на всех процессах! */
    A = (double*)malloc((size_t)N * N * sizeof(double));
    B = (double*)malloc((size_t)N * N * sizeof(double));
    C = (double*)malloc((size_t)N * N * sizeof(double));
    if (!A || !B || !C) {
        printf("Не хватает памяти (rank %d)\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    /* Инициализация матриц только на корневом процессе */
    if (rank == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i*N + j] = i + 1;
                B[i*N + j] = 1.0 / (j + 1);
                C[i*N + j] = 0.0;
            }
        }
    } else {
        /* На остальных можно просто обнулить C (не обязательно, но аккуратно) */
        for (i = 0; i < N*N; i++) {
            C[i] = 0.0;
        }
    }

    /* Рассылаем A и B всем процессам */
    MPI_Bcast(A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Разделяем строки между процессами */
    {
        int rows_per_process = N / size;
        int remainder = N % size;
        int extra = (rank < remainder) ? 1 : 0;
        int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
        int end_row = start_row + rows_per_process + extra;

        struct timespec ts_start, ts_end;

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        /* Локальное умножение: считаем строки [start_row, end_row) */
        for (i = start_row; i < end_row; i++) {
            for (j = 0; j < N; j++) {
                double s = 0.0;
                for (k = 0; k < N; k++) {
                    s += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = s;
            }
        }

        /* Сбор результата на rank 0 */
        if (rank != 0) {
            int rows_local = end_row - start_row;
            MPI_Send(&C[start_row * N], rows_local * N, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
        } else {
            int r;
            for (r = 1; r < size; r++) {
                int rp = rows_per_process + (r < remainder ? 1 : 0);
                int sr = r * rows_per_process + (r < remainder ? r : remainder);
                MPI_Recv(&C[sr * N], rp * N, MPI_DOUBLE,
                         r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);

        if (rank == 0) {
            double sec = (ts_end.tv_sec - ts_start.tv_sec)
                       + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

            printf("n = %d\n", N);
            printf("C[0][0] = %.6f\n", C[0]);
            printf("C[0][%d] = %.6f\n", N-1, C[N-1]);
            printf("C[%d][0] = %.6f\n", N-1, C[(N-1)*N + 0]);
            printf("C[%d][%d] = %.6f\n", N-1, N-1, C[(N-1)*N + (N-1)]);
            printf("Время: %.6f сек\n", sec);
            printf("Производительность: %.2f GFlops\n",
                   (2.0 * N * N * N) / (sec * 1e9));
        }
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
