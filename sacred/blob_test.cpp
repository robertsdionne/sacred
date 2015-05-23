#include <gtest/gtest.h>

#include "array.hpp"
#include "blob.hpp"

using sacred::Blob;

TEST(BlobTest, Initialize) {
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
