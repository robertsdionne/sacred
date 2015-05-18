#include <iostream>

#include "math.hpp"

int main(int argument_count, char *arguments[]) {
  constexpr auto K = 3, L = 3, M = 4, N = 4;
  float c[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };
  float a[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  float b[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  // float *C = new float[(M - K + 1) * (N - L + 1)], *A = new float[K * L], *B = new float[M * N];
  //
  // for (auto i = 0; i < (M - K + 1) * (N - L + 1); ++i) {
  //   C[i] = 0;
  // }
  //
  // for (auto i = 0; i < K * L; ++i) {
  //   A[i] = i;
  // }
  //
  // for (auto i = 0; i < M * N; ++i) {
  //   B[i] = i;
  // }

  sacred::Math<float> math;
  math.RecurrentConvolve2(c, a, 1.0, 1.0, M, N, K, L);
  math.Print(c, M, N);
  // math.Convolve2(c, a, b, 0.0, 1.0, K, L, M, N);
  // math.Print(c, M - K + 1, N - L + 1);
  // delete[] A;
  // delete[] B;
  // delete[] C;
  return 0;
}
