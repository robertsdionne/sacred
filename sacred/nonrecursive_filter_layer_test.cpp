#include <gtest/gtest.h>

#include "array.hpp"
#include "blob.hpp"
#include "nonrecursive_filter_layer.hpp"

using sacred::Array;
using sacred::Blob;
using sacred::NonrecursiveFilterLayer;

TEST(NonrecursiveFilterLayer, Forward) {
  auto input = Blob<float>({4, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  });
  auto bias = Blob<float>({2, 2});
  auto filter = Blob<float>({3, 3}, {
    9, 8, 7,
    6, 5, 4,
    3, 2, 1
  });
  auto layer = NonrecursiveFilterLayer<float>(bias, filter);
  auto output = Blob<float>({2, 2});
  layer.Forward(input, &output);
  EXPECT_EQ(
      Array<float>({2, 2}, {
        348, 393,
        528, 573
      }),
      output.value());
}
