#include <gtest/gtest.h>

#include "array.hpp"
#include "blob.hpp"
#include "dual.hpp"
#include "nonrecursive_filter_layer.hpp"
#include "recursive_filter_layer.hpp"

using namespace sacred;

TEST(Stacked, Forward) {
  auto input_0 = Blob<float>({8, 8}, {
    1, 2, 3, 4, 5, 6, 7, 8,
    2, 3, 4, 5, 6, 7, 8, 9,
    3, 4, 5, 6, 7, 8, 9, 10,
    4, 5, 6, 7, 8, 9, 10, 11,
    5, 6, 7, 8, 9, 10, 11, 12,
    6, 7, 8, 9, 10, 11, 12, 13,
    7, 8, 9, 10, 11, 12, 13, 14,
    8, 9, 10, 11, 12, 13, 14, 15
  });
  auto bias_0 = Blob<float>({6, 6});
  auto bias_1 = Blob<float>({6});
  auto filter = Blob<float>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    3, 4, 5
  });
  auto nonrecursive = NonrecursiveFilterLayer<float>(bias_0, filter);
  auto recursive = RecursiveFilterLayer<float>(bias_1, filter);
  auto input_1 = Blob<float>({6, 6});
  auto output = Blob<float>({6, 6});

  nonrecursive.Forward(input_0, &input_1);
  recursive.Forward(input_1, &output);

  std::cout << output.value() << std::endl;
}
