#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int rank;
    struct timespec ts_start, ts_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long msg_size = 10;
    long max_size = 10L * 1024 * 1024;

    if (rank == 0) {
        printf("# size(bytes)\tt_one_way(sec)\n");
    }

    while (msg_size <= max_size) {
        char *buffer = (char *)malloc(msg_size);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            clock_gettime(CLOCK_MONOTONIC, &ts_start);

            MPI_Send(buffer, (int)msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, (int)msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            clock_gettime(CLOCK_MONOTONIC, &ts_end);

            double sec = (ts_end.tv_sec - ts_start.tv_sec) +
                         (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

            double t_one_way = sec / 2.0;

            printf("%ld\t%g\n", msg_size, t_one_way);
        } else if (rank == 1) {
            MPI_Recv(buffer, (int)msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buffer, (int)msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }

        free(buffer);
        msg_size *= 8;
    }

    MPI_Finalize();
    return 0;
}
