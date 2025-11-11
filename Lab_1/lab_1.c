#include <stdio.h> //Подключает стандартный ввод/вывод (функции printf, scanf и т.д.)
#include <math.h> //Подключает математические функции и константы (например, M_PI, fabsl, sqrt и т.д.)
#include <stdlib.h> //Подключает общие утилиты: здесь используется atoi (преобразование строки в int) и exit/malloc и т.п.

int main(int argc, char *argv[]) { //Точка входа в программу. argc — количество аргументов командной строки, argv — массив строк (аргументы)

    int n = atoi(argv[1]); //Берёт первый аргумент командной строки argv[1] (строку) и преобразует в целое (int) функцией atoi

    if (n <= 0) {
        printf("n должно быть положительным");
        return 1;
    }

    long double sum = 0.0;

    int i;

    for (i = 1; i <= n; i++) {
        long double x = (i - 0.5L) / n;
        sum += 4.0L / (1.0L + x * x);
    }

    long double pi_approx = sum / n;
    long double pi_exact = M_PI;
    long double err = fabsl(pi_approx - pi_exact); //функция fabsl — абсолютное значение для long double

    printf("n = %d\n", n);
    printf("Точное значение pi = %.16Lf\n", pi_exact); //Печать «точного» значения π с форматом %.16Lf (16 знаков после запятой для long double)
    printf("Приблеженное значение pi = %.16Lf\n", pi_approx);
    printf("Разница = %.16Lf\n", err);
}