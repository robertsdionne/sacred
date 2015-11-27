#include <gtest/gtest.h>
#include <utility>

#include "dual.hpp"
#include "gradients.hpp"
#include "rectifier.hpp"
#include "rectifier_gradient.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(RectifierGradient, Run) {
  auto input = Tensor<>({3, 4}, {
    1, -2, 3, -4,
    -5, 6, -7, 8,
    9, -10, 11, -12,
  });
  auto input_gradient = Tensor<>({3, 4}, {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  });
  auto op = RectifierGradient<>();
  auto output_gradient = Tensor<>({3, 4}, {
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
  });

  op(output_gradient, input, input_gradient);

  EXPECT_EQ(Tensor<>({3, 4}, {
    1, 0, 1, 0,
    0, 1, 0, 1,
    1, 0, 1, 0,
  }), input_gradient);
}

TEST(RectifierGradient, Dual) {
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
