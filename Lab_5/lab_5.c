#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static void compute_block_decomposition(int N, int size, int rank, int *local_n, int *start)
{
    int base = N / size;
    int rem  = N % size;

    if (rank < rem) {
        *local_n = base + 1;
        *start   = rank * (*local_n);
    } else {
        *local_n = base;
        *start   = rem * (base + 1) + (rank - rem) * base;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Использование: %s N\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            fprintf(stderr, "N должен быть > 0\n");
        MPI_Finalize();
        return 1;
    }

    // Распределение строк матриц A и C
    int local_rows, row_start;
    compute_block_decomposition(N, size, rank, &local_rows, &row_start);

    // Распределение столбцов матрицы B (будем хранить B^T)
    int local_cols, col_start;
    compute_block_decomposition(N, size, rank, &local_cols, &col_start);

    // Выделение памяти 
    double *A_local = NULL;
    double *C_local = NULL;
    double *B_local_T = NULL;

    if (local_rows > 0) {
        A_local = (double *)malloc((size_t)local_rows * N * sizeof(double));
        C_local = (double *)malloc((size_t)local_rows * N * sizeof(double));
    }
    if (local_cols > 0) {
        B_local_T = (double *)malloc((size_t)local_cols * N * sizeof(double));
    }

    if ((local_rows > 0 && (!A_local || !C_local)) ||
        (local_cols > 0 && !B_local_T)) {
        fprintf(stderr, "Rank %d: не хватает памяти\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Инициализация A и B^T, обнуление C 
    int i, j, k;

    // A[i][j] = (i+1), только свои строки 
    for (i = 0; i < local_rows; ++i) {
        int ig = row_start + i;    // глобальный индекс строки
        for (j = 0; j < N; ++j) {
            A_local[i * N + j] = (double)(ig + 1);
        }
    }

    // B[k][j] = 1.0 / (j+1), но храним B^T: B_T[j][k] 
    for (j = 0; j < local_cols; ++j) {
        int jg = col_start + j;    // глобальный индекс столбца 
        for (k = 0; k < N; ++k) {
            B_local_T[j * N + k] = 1.0 / (double)(jg + 1);
        }
    }

    // C = 0 
    for (i = 0; i < local_rows; ++i)
        for (j = 0; j < N; ++j)
            C_local[i * N + j] = 0.0;

    // Буфер для принимаемых столбцов B (как вектор длины N)
    double *B_col_buf = NULL;
    if (N > 0) {
        B_col_buf = (double *)malloc((size_t)N * sizeof(double));
        if (!B_col_buf) {
            fprintf(stderr, "Rank %d: не хватает памяти на B_col_buf\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Основное умножение: C = A * B
    Стратегия:
    - матрица A распределена по строкам,
    - матрица B распределена по столбцам и хранится транспонированной,
    - по очереди каждый процесс "отдаёт" свои столбцы B всем через MPI_Bcast,
    - все процессы пересчитывают соответствующий столбец C для своих строк.
    */

    for (int owner = 0; owner < size; ++owner) {
        int owner_cols, owner_col_start;
        compute_block_decomposition(N, size, owner, &owner_cols, &owner_col_start);

        for (int jj = 0; jj < owner_cols; ++jj) {
            int jg = owner_col_start + jj;  // глобальный номер столбца B и C 

            double *buf;
            if (rank == owner) {
                // У владельца столбца уже есть нужный вектор в B_local_T 
                int j_local = jg - col_start;  // локальный индекс столбца на root
                buf = &B_local_T[j_local * N]; // B_T[j_local][0..N-1]
            } else {
                buf = B_col_buf;
            }

            // Передаем столбец B[:, jg] в виде вектора buf[k] остальным процессам
            MPI_Bcast(buf, N, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Считаем C[i][jg] = A[i][*] · B[*][jg] для своих строк i
            for (i = 0; i < local_rows; ++i) {
                double sum = 0.0;
                double *a_row = &A_local[i * N];
                for (k = 0; k < N; ++k) {
                    sum += a_row[k] * buf[k];
                }
                C_local[i * N + jg] = sum;
            }
        }
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Вычисление и печать нескольких элементов C
    double vals[4] = {0.0, 0.0, 0.0, 0.0};

    // C[0][0], C[0][N-1] 
    if (0 >= row_start && 0 < row_start + local_rows) {
        int il = 0 - row_start;
        vals[0] = C_local[il * N + 0];
        vals[1] = C_local[il * N + (N - 1)];
    }

    // C[N-1][0], C[N-1][N-1]
    if (N - 1 >= row_start && N - 1 < row_start + local_rows) {
        int il = (N - 1) - row_start;
        vals[2] = C_local[il * N + 0];
        vals[3] = C_local[il * N + (N - 1)];
    }

    double global_vals[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Reduce(vals, global_vals, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("n = %d, processes = %d\n", N, size);
        printf("C[0][0]     = %.6f\n", global_vals[0]);
        printf("C[0][%d]    = %.6f\n", N - 1, global_vals[1]);
        printf("C[%d][0]    = %.6f\n", N - 1, global_vals[2]);
        printf("C[%d][%d]   = %.6f\n", N - 1, N - 1, global_vals[3]);
        printf("Время: %.6f сек (максимум по процессам)\n", max_time);
        double gflops = 0.0;
        if (max_time > 0.0) {
            gflops = (2.0 * (double)N * (double)N * (double)N) / (max_time * 1e9);
        }
        printf("Производительность: %.2f GFlops\n", gflops);
    }

    free(A_local);
    free(C_local);
    free(B_local_T);
    free(B_col_buf);

    MPI_Finalize();
    return 0;
}
