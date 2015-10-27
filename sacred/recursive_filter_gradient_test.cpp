#include <gtest/gtest.h>
#include <iostream>

#include "dual.hpp"
#include "recursive_filter_gradient.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(RecursiveFilterGradient, Run) {
  auto output = Tensor<>({4, 4}, {
    1, 6, 36, 227,
    2, 14, 95, 635,
    3, 22, 157, 1093,
    4, 25, 171, 1196
  });
  auto output_gradient = Tensor<>({4, 4}, {
    0, 1, 2, 3,
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6
  });
  auto bias = Tensor<>({4});
  auto bias_gradient = Tensor<>({4});
  auto filter = Tensor<>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    4, 5, 6
  });
  auto filter_gradient = Tensor<>({3, 3});
  auto op = RecursiveFilterGradient<>(bias, filter, filter_gradient);
  auto input = Tensor<>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  auto input_gradient = Tensor<>({4, 4});
  op.Run({&output, &output_gradient}, {&input_gradient});
  EXPECT_EQ(Tensor<>({4}, {
    6, 10, 14, 18
  }), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3}, {
    6293, 573, 38,
    6475, 640, 50,
    4020, 410, 32
  }), filter_gradient);
  EXPECT_EQ(Tensor<>({4, 4}, {
    1926, 214, 24, 3,
    2288, 305, 34, 4,
    1675, 262, 42, 5,
    690, 118, 22, 6
  }), input_gradient);
}

// TEST(RecursiveFilterGradient, GradientInput) {
//   auto target = Tensor<Dual>({4, 4}, {
//     1, 6-1, 36-2, 227-3,
//     2-1, 14-2, 95-3, 635-4,
//     3-2, 22-3, 157-4, 1093-5,
//     4-3, 25-4, 171-5, 1196-6
//   });
//   auto error = Tensor<Dual>({4, 4});
//
//   for (auto k = 0; k < 4; ++k) {
//     for (auto l = 0; l < 4; ++l) {
//       auto input = Tensor<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         2, 3, 4, 5,
//         3, 4, 5, 6,
//         4, 5, 6, 7
//       });
//       input.value().axpby({k, l}, 1.0, 1_ɛ, 1.0);
//       auto bias = Tensor<Dual>({4});
//       auto filter = Tensor<Dual>({3, 3}, {
//         1, 2, 3,
//         2, 3, 4,
//         4, 5, 6
//       });
//       auto op = RecursiveFilterGradient<Dual>(bias, filter);
//       auto output = Tensor<Dual>({4, 4});
//       op.Forward(input, &output);
//
//       for (auto i = 0; i < 4; ++i) {
//         for (auto j = 0; j < 4; ++j) {
//           auto delta = target.at({i, j}) - output.value({i, j});
//           error.axpby({k, l}, 1.0, delta * delta / 2.0, 1.0);
//         }
//       }
//     }
//   }
//
//   EXPECT_EQ(Tensor<Dual>({4, 4}, {
//     92 + 1926_ɛ, 92 + 214_ɛ, 92 + 24_ɛ, 92 + 3_ɛ,
//     92 + 2288_ɛ, 92 + 305_ɛ, 92 + 34_ɛ, 92 + 4_ɛ,
//     92 + 1675_ɛ, 92 + 262_ɛ, 92 + 42_ɛ, 92 + 5_ɛ,
//     92 + 690_ɛ, 92 + 118_ɛ, 92 + 22_ɛ, 92 + 6_ɛ
//   }), error);
// }
//
// TEST(RecursiveFilterGradient, GradientFilter) {
//   auto target = Tensor<Dual>({4, 4}, {
//     1, 6-1, 36-2, 227-3,
//     2-1, 14-2, 95-3, 635-4,
//     3-2, 22-3, 157-4, 1093-5,
//     4-3, 25-4, 171-5, 1196-6
//   });
//   auto error = Tensor<Dual>({3, 3});
//
//   for (auto k = 0; k < 3; ++k) {
//     for (auto l = 0; l < 3; ++l) {
//       auto input = Tensor<Dual>({4, 4}, {
//         1, 2, 3, 4,
//         2, 3, 4, 5,
//         3, 4, 5, 6,
//         4, 5, 6, 7
//       });
//       auto bias = Tensor<Dual>({4});
//       auto filter = Tensor<Dual>({3, 3}, {
//         1, 2, 3,
//         2, 3, 4,
//         4, 5, 6
//       });
//       filter.value().axpby({k, l}, 1.0, 1_ɛ, 1.0);
//       auto op = RecursiveFilterGradient<Dual>(bias, filter);
//       auto output = Tensor<Dual>({4, 4});
//       op.Forward(input, &output);
//
//       for (auto i = 0; i < 4; ++i) {
//         for (auto j = 0; j < 4; ++j) {
//           auto delta = target.at({i, j}) - output.value({i, j});
//           error.axpby({k, l}, 1.0, delta * delta / 2.0, 1.0);
//         }
//       }
//     }
//   }
//
//   EXPECT_EQ(Tensor<Dual>({3, 3}, {
//     92 + 6293_ɛ, 92 + 573_ɛ, 92 + 38_ɛ,
//     92 + 6475_ɛ, 92 + 640_ɛ, 92 + 50_ɛ,
//     92 + 4020_ɛ, 92 + 410_ɛ, 92 + 32_ɛ
//   }), error);
// }
//
// TEST(RecursiveFilterGradient, GradientBias) {
//   auto target = Tensor<Dual>({4, 4}, {
//     1, 6-1, 36-2, 227-3,
//     2-1, 14-2, 95-3, 635-4,
//     3-2, 22-3, 157-4, 1093-5,
//     4-3, 25-4, 171-5, 1196-6
//   });
//   auto error = Tensor<Dual>({4});
//
//   for (auto k = 0; k < 4; ++k) {
//     auto input = Tensor<Dual>({4, 4}, {
//       1, 2, 3, 4,
//       2, 3, 4, 5,
//       3, 4, 5, 6,
//       4, 5, 6, 7
//     });
//     auto bias = Tensor<Dual>({4});
//     bias.value().axpby({k}, 1.0, 1_ɛ, 1.0);
//     auto filter = Tensor<Dual>({3, 3}, {
//       1, 2, 3,
//       2, 3, 4,
//       4, 5, 6
//     });
//     auto op = RecursiveFilterGradient<Dual>(bias, filter);
//     auto output = Tensor<Dual>({4, 4});
//     op.Forward(input, &output);
//
//     for (auto i = 0; i < 4; ++i) {
//       for (auto j = 0; j < 4; ++j) {
//         auto delta = target.at({i, j}) - output.value({i, j});
//         error.axpby({k}, 1.0, delta * delta / 2.0, 1.0);
//       }
//     }
//   }
//
//   EXPECT_EQ(Tensor<Dual>({4}, {
//     92 + 6_ɛ, 92 + 10_ɛ, 92 + 14_ɛ, 92 + 18_ɛ
//   }), error);
// }

}  // namespace sacred
