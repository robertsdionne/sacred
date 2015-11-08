#include <gtest/gtest.h>
#include <tuple>
#include <utility>

#include "dual.hpp"
#include "fully_connected.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(FullyConnected, Run) {
  auto input = Tensor<>({4, 1}, {1, 2, 3, 4});
  auto bias = Tensor<>({3, 1}, {1, 2, 3});
  auto weight = Tensor<>({3, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
  });
  auto op = FullyConnected<>(bias, weight);
  auto output = Tensor<>({3, 1}, {0, 0, 0});

  op(input, output);

  EXPECT_EQ(Tensor<>({3, 1}, {31, 72, 113}), output);
}

TEST(FullyConnected, Gradient) {
  using std::make_pair;
  using std::tie;

  auto input = Tensor<Dual>({4, 1}, {1, 2, 3, 4});
  auto expected_input_gradient = Tensor<>({4, 1}, {38, 44, 50, 56});

  auto bias = Tensor<Dual>({3, 1}, {1, 2, 3});
  auto expected_bias_gradient = Tensor<>({3, 1}, {1, 2, 3});

  auto weight = Tensor<Dual>({3, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
  });
  auto expected_weight_gradient = Tensor<>({3, 4}, {
    1, 2, 3, 4,
    2, 4, 6, 8,
    3, 6, 9, 12,
  });

  auto target = Tensor<Dual>({3, 1}, {30, 70, 110});

  for (auto pair : {
    make_pair(&input, &expected_input_gradient),
    make_pair(&bias, &expected_bias_gradient),
    make_pair(&weight, &expected_weight_gradient),
  }) {
    Tensor<Dual> *parameter;
    Tensor<> *expected_parameter_gradient;
    tie(parameter, expected_parameter_gradient) = pair;

    for (auto i = 0; i < parameter->shape().at(0); ++i) {
      for (auto j = 0; j < parameter->shape().at(1); ++j) {
        auto output = Tensor<Dual>({3, 1}, {0, 0, 0});
        auto op = FullyConnected<Dual>(bias, weight);

        parameter->at({i, j}) += 1_ɛ;

        op(input, output);

        auto loss = 0_ɛ;
        for (auto u = 0; u < target.shape().at(0); ++u) {
          for (auto v = 0; v < target.shape().at(1); ++v) {
            auto delta = target.at({u, v}) - output.at({u, v});
            loss += delta * delta / 2.0f;
          }
        }

        EXPECT_EQ(expected_parameter_gradient->at({i, j}), loss.dual);

        parameter->at({i, j}).dual = 0;
      }
    }
  }
}

}  // namespace sacred
