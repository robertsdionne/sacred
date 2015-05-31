#include <gtest/gtest.h>

#include "array.hpp"
#include "blob.hpp"

using sacred::Blob;

TEST(Blob, Initialize) {
  auto blob = Blob<float>({1, 2, 3});
  EXPECT_EQ(3, blob.number_of_axes());
  EXPECT_EQ(6, blob.count());

  EXPECT_EQ(1, blob.shape(0));
  EXPECT_EQ(2, blob.shape(1));
  EXPECT_EQ(3, blob.shape(2));

  EXPECT_EQ(1, blob.value().shape(0));
  EXPECT_EQ(2, blob.value().shape(1));
  EXPECT_EQ(3, blob.value().shape(2));

  EXPECT_EQ(1, blob.diff().shape(0));
  EXPECT_EQ(2, blob.diff().shape(1));
  EXPECT_EQ(3, blob.diff().shape(2));

  blob.Reshape({2, 3, 4});
  EXPECT_EQ(24, blob.count());

  EXPECT_EQ(2, blob.shape(0));
  EXPECT_EQ(3, blob.shape(1));
  EXPECT_EQ(4, blob.shape(2));

  EXPECT_EQ(2, blob.value().shape(0));
  EXPECT_EQ(3, blob.value().shape(1));
  EXPECT_EQ(4, blob.value().shape(2));

  EXPECT_EQ(2, blob.diff().shape(0));
  EXPECT_EQ(3, blob.diff().shape(1));
  EXPECT_EQ(4, blob.diff().shape(2));
}

TEST(Blob, InitializeValue) {
  auto blob = Blob<float>({1, 2, 3}, {0, 1, 2, 3, 4, 5});

  for (auto j = 0; j < 2; ++j) {
    for (auto k = 0; k < 3; ++k) {
      EXPECT_EQ(3 * j + k, blob.value().at({0, j, k}));
      EXPECT_EQ(0, blob.diff().at({0, j, k}));
    }
  }

  blob = Blob<float>({1, 2, 3}, {0, 1, 2, 3, 4, 5}, {0, -1, -2, -3, -4, -5});

  for (auto j = 0; j < 2; ++j) {
    for (auto k = 0; k < 3; ++k) {
      EXPECT_EQ(3 * j + k, blob.value().at({0, j, k}));
      EXPECT_EQ(-3 * j - k, blob.diff().at({0, j, k}));
    }
  }
}
