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
  auto bias = Tensor<Dual>({3, 1}, {1, 2, 3});
  auto weight = Tensor<Dual>({3, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
  });
  auto target = Tensor<Dual>({3, 1}, {30, 70, 110});

  auto input_gradient = Tensor<>({4, 1});
  auto bias_gradient = Tensor<>({3, 1});
  auto weight_gradient = Tensor<>({3, 4});

  for (auto pair : {
    make_pair(&input, &input_gradient),
    make_pair(&bias, &bias_gradient),
    make_pair(&weight, &weight_gradient),
  }) {
    Tensor<Dual> *parameter;
    Tensor<> *parameter_gradient;
    tie(parameter, parameter_gradient) = pair;

    for (auto i = 0; i < parameter->size(); ++i) {
      auto output = Tensor<Dual>({3, 1}, {0, 0, 0});
      auto op = FullyConnected<Dual>(bias, weight);

      parameter->data(i) += 1_ɛ;

      op(input, output);

      auto loss = 0_ɛ;
      for (auto j = 0; j < target.size(); ++j) {
        auto delta = target.data(j) - output.data(j);
        loss += delta * delta / 2.0f;
      }

      parameter_gradient->data(i) = loss.dual;

      parameter->data(i).dual = 0;
    }
  }

  EXPECT_EQ(Tensor<>({4, 1}, {38, 44, 50, 56}), input_gradient);
  EXPECT_EQ(Tensor<>({3, 1}, {1, 2, 3}), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 4}, {
    1, 2, 3, 4,
    2, 4, 6, 8,
    3, 6, 9, 12,
  }), weight_gradient);
}

}  // namespace sacred
