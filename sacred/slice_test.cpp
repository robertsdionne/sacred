#include <gtest/gtest.h>

#include "slice.hpp"

using sacred::Slice;

constexpr auto kN = 10;

TEST(Slice, InitializePositiveStep) {
  auto slice = Slice();

  EXPECT_FALSE(slice.has_start());
  EXPECT_FALSE(slice.has_stop());

  EXPECT_EQ(0, slice.start(kN));
  EXPECT_EQ(kN, slice.stop(kN));
  EXPECT_EQ(1, slice.step());

  slice = Slice(1);

  EXPECT_TRUE(slice.has_start());
  EXPECT_FALSE(slice.has_stop());

  EXPECT_EQ(1, slice.start(kN));
  EXPECT_EQ(kN, slice.stop(kN));
  EXPECT_EQ(1, slice.step());

  slice = Slice(1, 2);

  EXPECT_TRUE(slice.has_start());
  EXPECT_TRUE(slice.has_stop());

  EXPECT_EQ(1, slice.start(kN));
  EXPECT_EQ(2, slice.stop(kN));
  EXPECT_EQ(1, slice.step());

  slice = Slice(1, 2, 3);

  EXPECT_TRUE(slice.has_start());
  EXPECT_TRUE(slice.has_stop());

  EXPECT_EQ(1, slice.start(kN));
  EXPECT_EQ(2, slice.stop(kN));
  EXPECT_EQ(3, slice.step());

  slice = Slice(nullptr, 2);

  EXPECT_FALSE(slice.has_start());
  EXPECT_TRUE(slice.has_stop());

  EXPECT_EQ(0, slice.start(kN));
  EXPECT_EQ(2, slice.stop(kN));
  EXPECT_EQ(1, slice.step());

  slice = Slice(1, nullptr, 3);

  EXPECT_TRUE(slice.has_start());
  EXPECT_FALSE(slice.has_stop());

  EXPECT_EQ(1, slice.start(kN));
  EXPECT_EQ(kN, slice.stop(kN));
  EXPECT_EQ(3, slice.step());

  slice = Slice(nullptr, 2, 3);

  EXPECT_FALSE(slice.has_start());
  EXPECT_TRUE(slice.has_stop());

  EXPECT_EQ(0, slice.start(kN));
  EXPECT_EQ(2, slice.stop(kN));
  EXPECT_EQ(3, slice.step());

  slice = Slice(nullptr, nullptr, 3);

  EXPECT_FALSE(slice.has_start());
  EXPECT_FALSE(slice.has_stop());

  EXPECT_EQ(0, slice.start(kN));
  EXPECT_EQ(kN, slice.stop(kN));
  EXPECT_EQ(3, slice.step());
}

TEST(Slice, InitializeNegativeStep) {
  auto slice = Slice(1, 2, -3);

  EXPECT_TRUE(slice.has_start());
  EXPECT_TRUE(slice.has_stop());

  EXPECT_EQ(1, slice.start(kN));
  EXPECT_EQ(2, slice.stop(kN));
  EXPECT_EQ(-3, slice.step());

  slice = Slice(1, nullptr, -3);

  EXPECT_TRUE(slice.has_start());
  EXPECT_FALSE(slice.has_stop());

  EXPECT_EQ(1, slice.start(kN));
  EXPECT_EQ(-1, slice.stop(kN));
  EXPECT_EQ(-3, slice.step());

  slice = Slice(nullptr, 2, -3);

  EXPECT_FALSE(slice.has_start());
  EXPECT_TRUE(slice.has_stop());

  EXPECT_EQ(kN, slice.start(kN));
  EXPECT_EQ(2, slice.stop(kN));
  EXPECT_EQ(-3, slice.step());

  slice = Slice(nullptr, nullptr, -3);

  EXPECT_FALSE(slice.has_start());
  EXPECT_FALSE(slice.has_stop());

  EXPECT_EQ(kN, slice.start(kN));
  EXPECT_EQ(-1, slice.stop(kN));
  EXPECT_EQ(-3, slice.step());
}
