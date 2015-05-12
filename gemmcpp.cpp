#include <iostream>

/**
 * Multiplies matrices A and B to produce the result C.
 *
 * In Einstein notation: C^i_j = beta * C^i_j + alpha * A^{ik} * B_{kj}
 */
template<int M, int N, int K>
void Gemm(
    float c[M][N], const float a[M][K], const float b[K][N], const float alpha, const float beta) {
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      c[i][j] *= beta;
      for (auto k = 0; k < K; ++k) {
        c[i][j] += alpha * a[i][k] * b[k][j];
      }
    }
  }
}

template <int M, int N>
void Print(const float a[M][N]) {
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      std::cout << a[i][j] << u8" ";
    }
    std::cout << std::endl;
  }
}

int main(int argument_count, char *arguments[]) {
  constexpr auto M = 4, N = 3, K = 2;
  float c[M][N] = {0};
  float a[M][K] = {
    {1, 2},
    {3, 4},
    {5, 6},
    {7, 8}
  };
  float b[K][N] = {
    {1, 2, 3},
    {4, 5, 6}
  };
  Gemm<M, N, K>(c, a, b, 1.0, 0.0);
  Gemm<M, N, K>(c, a, b, 1.0, 0.0);
  Print<M, N>(c);
  return 0;
}
