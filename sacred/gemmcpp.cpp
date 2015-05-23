#include <iostream>

#include "layer.hpp"
#include "math.hpp"

int main(int argument_count, char *arguments[]) {
  constexpr auto M = 4, N = 3, K = 2;
  float c[] = {
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
  };
  float a[] = {
    1, 2,
    3, 4,
    5, 6,
    7, 8
  };
  float b[] = {
    1, 2, 3,
    4, 5, 6
  };
  sacred::Math<float> math;
  math.GeneralMatrixMultiplication(c, a, b, 0.0, 1.0, M, N, K);
  math.Print(c, M, N);
  return 0;
}
