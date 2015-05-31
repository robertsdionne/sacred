#include <gtest/gtest.h>
#include <iostream>

#include "array.hpp"
#include "blob.hpp"
#include "dual.hpp"
#include "recursive_filter_layer.hpp"

using namespace sacred;

TEST(RecursiveFilterLayer, Forward) {
  auto input = Blob<float>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  auto bias = Blob<float>({4});
  auto filter = Blob<float>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    4, 5, 6
  });
  auto layer = RecursiveFilterLayer<float>(bias, filter);
  auto output = Blob<float>({4, 4});
  layer.Forward(input, &output);
  EXPECT_EQ(
      Array<float>({4, 4}, {
        1, 6, 36, 227,
        2, 14, 95, 635,
        3, 22, 157, 1093,
        4, 25, 171, 1196
      }),
      output.value());
}

TEST(RecursiveFilterLayer, BackwardFilter) {
  auto target = Array<Dual>({4, 4}, {
    1, 6-1, 36-2, 227-3,
    2-1, 14-2, 95-3, 635-4,
    3-2, 22-3, 157-4, 1093-5,
    4-3, 25-4, 171-5, 1196-6
  });
  auto error = Array<Dual>({3, 3});

  for (auto k = 0; k < 3; ++k) {
    for (auto l = 0; l < 3; ++l) {
      auto input = Blob<Dual>({4, 4}, {
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7
      });
      auto bias = Blob<Dual>({4});
      auto filter = Blob<Dual>({3, 3}, {
        1, 2, 3,
        2, 3, 4,
        4, 5, 6
      });
      filter.value({k, l}) += 1_ɛ;
      auto layer = RecursiveFilterLayer<Dual>(bias, filter);
      auto output = Blob<Dual>({4, 4});
      layer.Forward(input, &output);

      for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 4; ++j) {
          auto delta = target.at({i, j}) - output.value({i, j});
          error.at({k, l}) += delta * delta / 2.0;
        }
      }
    }
  }

  EXPECT_EQ(Array<Dual>({3, 3}, {
    92 + 6293_ɛ, 92 + 573_ɛ, 92 + 38_ɛ,
    92 + 6475_ɛ, 92 + 640_ɛ, 92 + 50_ɛ,
    92 + 4020_ɛ, 92 + 410_ɛ, 92 + 32_ɛ
  }), error);
}

// TEST(RecursiveFilterLayer, Backward) {
//   auto output = Blob<float>({4, 4}, {
//     1, 53, 1446, 34685,
//     5, 120, 2749, 48191,
//     9, 196, 2777, 39825,
//     13, 119, 1400, 17795
//   }, {
//     1, -1, 1, 1,
//     -1, 1, -1, -1,
//     1, 1, -1, 1,
//     -1, -1, -1, +1
//   });
//   auto bias = Blob<float>({4, 1});
//   auto filter = Blob<float>({3, 3}, {
//     9, 8, 7,
//     6, 5, 4,
//     3, 2, 1
//   });
//   auto layer = RecursiveFilterLayer<float>(bias, filter);
//   auto input = Blob<float>({4, 4}, {
//     1, 2, 3, 4,
//     5, 6, 7, 8,
//     9, 10, 11, 12,
//     13, 14, 15, 16
//   });
//   layer.Backward(output, &input);
//   // TODO(robertsdionne): properly determine these values.
//   EXPECT_EQ(Array<float>({4, 1}, {
//     2, -2, 2, -2
//   }), bias.diff());
//   EXPECT_EQ(Array<float>({3, 3}, {
//     -20653, 20730, 1296,
//     10801, -7958, -1224,
//     54805, -49122, -3812
//   }), filter.diff());
//   EXPECT_EQ(Array<float>({4, 4}, {
//     487, 41, 4, 1,
//     1084, 69, 5, -1,
//     1430, 81, -1, 1,
//     1323, 87, 14, 1
//   }), input.diff());
// }
//
// TEST(RecursiveFilterLayer, Gradient) {
//   auto target = Blob<Dual>({4, 4}, {
//     1+1, 53-1, 1446+1, 34685+1,
//     5-1, 120+1, 2749-1, 48191-1,
//     9+1, 196+1, 2777-1, 39825+1,
//     13-1, 119-1, 1400-1, 17795+1
//   });
//
//   for (auto i = 0; i < 4; ++i) {
//     for (auto j = 0; j < 4; ++j) {
//       auto input = Blob<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9, 10, 11, 12,
//         13, 14, 15, 16
//       });
//       auto bias = Blob<Dual>({4, 1});
//       auto filter = Blob<Dual>({3, 3}, {
//         9, 8, 7,
//         6, 5, 4,
//         3, 2, 1
//       });
//       auto layer = RecursiveFilterLayer<Dual>(bias, filter);
//       auto output = Blob<Dual>({4, 4});
//
//       input.value({i, j}).dual = 1.0;
//       layer.Forward(input, &output);
//
//       // std::cout << output.value() << std::endl;
//
//       auto loss = 0_ɛ;
//       for (auto k = 0; k < output.shape(0); ++k) {
//         for (auto l = 0; l < output.shape(1); ++l) {
//           auto delta = target.value({k, l}) - output.value({k, l});
//           output.diff({k, l}) = -delta;
//           loss += delta * delta / 2.0f;
//         }
//       }
//
//       // layer.Backward(output, &input);
//       input.value({i, j}).dual = 0.0;
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
//       auto input = Blob<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9, 10, 11, 12,
//         13, 14, 15, 16
//       });
//       auto bias = Blob<Dual>({4, 1});
//       auto filter = Blob<Dual>({3, 3}, {
//         9, 8, 7,
//         6, 5, 4,
//         3, 2, 1
//       });
//       auto layer = RecursiveFilterLayer<Dual>(bias, filter);
//       auto output = Blob<Dual>({4, 4});
//
//       filter.value({i, j}).dual = 1.0;
//       layer.Forward(input, &output);
//
//       std::cout << output.value() << std::endl;
//
//       auto loss = 0_ɛ;
//       for (auto k = 0; k < output.shape(0); ++k) {
//         for (auto l = 0; l < output.shape(1); ++l) {
//           auto delta = target.value({k, l}) - output.value({k, l});
//           output.diff({k, l}) = -delta;
//           loss += delta * delta / 2.0f;
//         }
//       }
//
//       // layer.Backward(output, &input);
//       filter.value({i, j}).dual = 0.0;
//
//       auto actual_partial_error_with_respect_to_filter_ij = filter.diff({i, j}).real;
//       auto expected_partial_error_with_respect_to_filter_ij = loss.dual;
//
//       EXPECT_NEAR(expected_partial_error_with_respect_to_filter_ij,
//           actual_partial_error_with_respect_to_filter_ij, 1e-1);
//     }
//   }
//
//   for (auto i = 0; i < 4; ++i) {
//     auto input = Blob<Dual>({4, 4}, {
//       1, 2, 3, 4,
//       5, 6, 7, 8,
//       9, 10, 11, 12,
//       13, 14, 15, 16
//     });
//     auto bias = Blob<Dual>({4, 1});
//     auto filter = Blob<Dual>({3, 3}, {
//       9, 8, 7,
//       6, 5, 4,
//       3, 2, 1
//     });
//     auto layer = RecursiveFilterLayer<Dual>(bias, filter);
//     auto output = Blob<Dual>({4, 4});
//
//     bias.value(i).dual = 1.0;
//     layer.Forward(input, &output);
//
//     auto loss = 0_ɛ;
//     for (auto k = 0; k < output.shape(0); ++k) {
//       for (auto l = 0; l < output.shape(1); ++l) {
//         auto delta = target.value({k, l}) - output.value({k, l});
//         output.diff({k, l}) = -delta;
//         loss += delta * delta / 2.0f;
//       }
//     }
//
//     // layer.Backward(output, &input);
//     bias.value(i).dual = 0.0;
//
//     auto actual_partial_error_with_respect_to_bias_i = bias.diff(i).real;
//     auto expected_partial_error_with_respect_to_bias_i = loss.dual;
//
//     EXPECT_NEAR(expected_partial_error_with_respect_to_bias_i,
//         actual_partial_error_with_respect_to_bias_i, 1e-1);
//   }
// }
