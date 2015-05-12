#include <iostream>

/**
 * Multiplies matrices A and B to produce the result C.
 *
 * In Einstein notation: C^i_j = beta * C^i_j + alpha * A^{ik} * B_{kj}
 */
template<int M, int N, int K>
void Gemm(float *c, const float *a, const float *b, const float alpha, const float beta) {
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      c[i * N + j] *= beta;
      for (auto k = 0; k < K; ++k) {
        c[i * N + j] += alpha * a[i * K + k] * b[k * N + j];
      }
    }
  }
}

template <int M, int N>
void Print(const float *a) {
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      std::cout << a[i * N + j] << u8" ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/**
 * Convolves a filter A with matrix B to produce the result C.
 *
 * C = A ∗ B
 */
template <int K, int L, int M, int N>
void Conv(float *c, const float *a, const float *b) {
  const auto Y = K * L;
  const auto Z = (M - K + 1) * (N - L + 1);
  float *d = new float[Y * Z];
  for (auto i = 0; i < M - K + 1; ++i) {
    for (auto j = 0; j < N - L + 1; ++j) {
      for (auto k = 0; k < K; ++k) {
        for (auto l = 0; l < L; ++l) {
          auto f = k * L + l;
          auto g = i * (N - L + 1) + j;
          d[f * (M - K + 1) * (N - L + 1) + g] = b[(i + k) * N + (j + l)];
        }
      }
    }
  }
  Gemm<1, Z, Y>(c, a, d, 1.0, 0.0);
  delete[] d;
}

int main(int argument_count, char *arguments[]) {
  constexpr auto K = 2, L = 2, M = 4, N = 4;
  float c[] = {
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
  };
  float a[] = {
    1, 2,
    3, 4
  };
  float b[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };
  Conv<K, L, M, N>(c, a, b);
  Print<M - K + 1, N - L + 1>(c);
  return 0;
}
