#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    int n = atoi(argv[1]);

    if (n <= 0) {
        printf("n должно быть положительным");
        return 1;
    }

    const int NUM_THREADS = 5;
    omp_set_num_threads(NUM_THREADS);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    long double sum = 0.0;
    int i;
    
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (i = 1; i <= n; i++) {
        long double x = (i - 0.5L) / n;
        sum += 4.0L / (1.0L + x * x);
    }

    long double pi_approx = sum / n;
    long double pi_exact = M_PI;
    long double err = fabsl(pi_approx - pi_exact);

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double sec = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("Итерации (n) = %d\n", n);
    printf("Точное значение pi = %.16Lf\n", pi_exact);
    printf("Приблеженное значение pi = %.16Lf\n", pi_approx);
    printf("Разница = %.16Lf\n", err);
    printf("Потоков = %d\n", NUM_THREADS);
    printf("Время = %.6f сек\n", sec);
}
