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

template <int K, int L, int M, int N>
void Conv2(float *c, const float *a, const float *b) {
  for (auto i = 0; i < M - K + 1; ++i) {
    for (auto j = 0; j < N - L + 1; ++j) {
      for (auto k = 0; k < K; ++k) {
        for (auto l = 0; l < L; ++l) {
          c[i * (N - L + 1) + j] += a[k * L + l] * b[(i + k) * N + (j + l)];
        }
      }
    }
  }
}

template <int K, int L, int M, int N>
void ReConv2(float *c, const float *a, const float alpha, const float beta) {
  for (auto j = 0; j < N; ++j) {
    for (auto i = 0; i < M; ++i) {
      float c_out = beta * c[i * N + j];
      for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
          int x = j + l - L;
          int y = i + k - K / 2;
          bool in = 0 <= x && 0 <= y && y < M;
          if (in) {
            c_out += alpha * a[k * L + l] * c[y * N + x];
          }
        }
      }
      c[i * N + j] = c_out;
    }
  }
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

  ReConv2<K, L, M, N>(c, a, 1.0, 1.0);
  Print<M, N>(c);
  // delete[] A;
  // delete[] B;
  // delete[] C;
  return 0;
}
