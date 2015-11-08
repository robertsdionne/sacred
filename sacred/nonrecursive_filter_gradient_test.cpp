#include <gtest/gtest.h>

#include "dual.hpp"
#include "nonrecursive_filter_gradient.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(NonrecursiveFilterGradient, Run) {
  auto output = Tensor<>({2, 2}, {
    69, 96,
    96, 123
  });
  auto output_gradient = Tensor<>({2, 2}, {
    1, 1,
    1, 1
  });
  auto bias = Tensor<>({2, 2});
  auto bias_gradient = Tensor<>({2, 2});
  auto filter = Tensor<>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    3, 4, 5
  });
  auto filter_gradient = Tensor<>({3, 3});
  auto op = NonrecursiveFilterGradient<>(bias_gradient, filter, filter_gradient);
  auto input = Tensor<>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  auto input_gradient = Tensor<>({4, 4});

  op(output_gradient, input, input_gradient);

  EXPECT_EQ(Tensor<>({2, 2}, {
    1, 1,
    1, 1
  }), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3}, {
    24, 20, 16,
    20, 16, 12,
    16, 12, 8
  }), filter_gradient);
  EXPECT_EQ(Tensor<>({4, 4}, {
    5, 9, 7, 3,
    9, 16, 12, 5,
    7, 12, 8, 3,
    3, 5, 3, 1
  }), input_gradient);
}

// TEST(NonrecursiveFilterGradient, Gradient) {
//   auto target = Tensor<Dual>({2, 2}, {
//     69, 96,
//     96, 123
//   });
//
//   for (auto i = 0; i < 4; ++i) {
//     for (auto j = 0; j < 4; ++j) {
//       auto input = Tensor<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         2, 3, 4, 5,
//         3, 4, 5, 6,
//         4, 5, 6, 7
//       });
//       auto bias = Tensor<Dual>({2, 2});
//       auto filter = Tensor<Dual>({3, 3}, {
//         1, 2, 3,
//         2, 3, 4,
//         3, 4, 5
//       });
//       auto op = NonrecursiveFilterGradient<Dual>(bias, filter);
//       auto output = Tensor<Dual>({2, 2});
//
//       input.value().add({i, j}, 1_ɛ);
//       op.Forward(input, &output);
//
//       auto loss = 0_ɛ;
//       for (auto k = 0; k < output.shape(0); ++k) {
//         for (auto l = 0; l < output.shape(1); ++l) {
//           auto delta = target.value({k, l}) - output.value({k, l});
//           output.diff().set({k, l}, -delta);
//           loss += delta * delta / 2.0f;
//         }
//       }
//
//       op.Backward(output, &input);
//
//       auto actual_partial_error_with_respect_to_input_ij = input.diff({i, j}).real;
//       auto expected_partial_error_with_respect_to_input_ij = loss.dual;
//
//       EXPECT_NEAR(expected_partial_error_with_respect_to_input_ij, actual_partial_error_with_respect_to_input_ij, 1e-1);
//     }
//   }
//
//   for (auto i = 0; i < 3; ++i) {
//     for (auto j = 0; j < 3; ++j) {
//       auto input = Tensor<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         2, 3, 4, 5,
//         3, 4, 5, 6,
//         4, 5, 6, 7
//       });
//       auto bias = Tensor<Dual>({2, 2});
//       auto filter = Tensor<Dual>({3, 3}, {
//         1, 2, 3,
//         2, 3, 4,
//         3, 4, 5
//       });
//       auto op = NonrecursiveFilterGradient<Dual>(bias, filter);
//       auto output = Tensor<Dual>({2, 2});
//
//       filter.value().add({i, j}, 1_ɛ);
//       op.Forward(input, &output);
//
//       auto loss = 0_ɛ;
//       for (auto k = 0; k < output.shape(0); ++k) {
//         for (auto l = 0; l < output.shape(1); ++l) {
//           auto delta = target.value({k, l}) - output.value({k, l});
//           output.diff().set({k, l}, -delta);
//           loss += delta * delta / 2.0f;
//         }
//       }
//
//       op.Backward(output, &input);
//
//       auto actual_partial_error_with_respect_to_filter_ij = filter.diff({i, j}).real;
//       auto expected_partial_error_with_respect_to_filter_ij = loss.dual;
//
//       EXPECT_NEAR(expected_partial_error_with_respect_to_filter_ij,
//           actual_partial_error_with_respect_to_filter_ij, 1e-1);
//     }
//   }
//
//   for (auto i = 0; i < 2; ++i) {
//     for (auto j = 0; j < 2; ++j) {
//       auto input = Tensor<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         2, 3, 4, 5,
//         3, 4, 5, 6,
//         4, 5, 6, 7
//       });
//       auto bias = Tensor<Dual>({2, 2});
//       auto filter = Tensor<Dual>({3, 3}, {
//         1, 2, 3,
//         2, 3, 4,
//         3, 4, 5
//       });
//       auto op = NonrecursiveFilterGradient<Dual>(bias, filter);
//       auto output = Tensor<Dual>({2, 2});
//
//       bias.value().add({i, j}, 1_ɛ);
//       op.Forward(input, &output);
//
//       auto loss = 0_ɛ;
//       for (auto k = 0; k < output.shape(0); ++k) {
//         for (auto l = 0; l < output.shape(1); ++l) {
//           auto delta = target.value({k, l}) - output.value({k, l});
//           output.diff().set({k, l}, -delta);
//           loss += delta * delta / 2.0f;
//         }
//       }
//
//       op.Backward(output, &input);
//
//       auto actual_partial_error_with_respect_to_bias_ij = bias.diff({i, j}).real;
//       auto expected_partial_error_with_respect_to_bias_ij = loss.dual;
//
//       EXPECT_NEAR(expected_partial_error_with_respect_to_bias_ij,
//           actual_partial_error_with_respect_to_bias_ij, 1e-1);
//     }
//   }
// }

}  // namespace sacred
