#include <gtest/gtest.h>
#include <utility>

#include "dual.hpp"
#include "gradients.hpp"
#include "rectifier.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(Rectifier, Run) {
  auto input = Tensor<>({3, 4}, {
    1, -2, 3, -4,
    -5, 6, -7, 8,
    9, -10, 11, -12,
  });
  auto op = Rectifier<>();
  auto output = Tensor<>({3, 4}, {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  });

  op(input, output);

  EXPECT_EQ(Tensor<>({3, 4}, {
    1, 0, 3, 0,
    0, 6, 0, 8,
    9, 0, 11, 0,
  }), output);
}

TEST(Rectifier, Gradient) {
  using std::make_pair;

  auto input = Tensor<Dual>({3, 4}, {
    1, -2, 3, -4,
    -5, 6, -7, 8,
    9, -10, 11, -12,
  });
  auto target = Tensor<Dual>({3, 4}, {
    0, -1, 2, -1,
    -1, 5, -1, 7,
    8, -1, 10, -1,
  });

  auto input_gradient = Tensor<>({3, 4});

  TestGradients<Rectifier<Dual>>({
    make_pair(&input, &input_gradient),
  }, [] () {
    return new Tensor<Dual>({3, 4}, {
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
    });
  }, [] () {
    return new Rectifier<Dual>();
  }, input, target);

  EXPECT_EQ(Tensor<>({3, 4}, {
    1, 0, 1, 0,
    0, 1, 0, 1,
    1, 0, 1, 0,
  }), input_gradient);
}

}  // namespace sacred
