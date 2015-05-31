#include <gtest/gtest.h>

#include "array.hpp"

using sacred::Array;

TEST(Array, Initialize1D) {
  auto array = Array<float>({3}, {1, 2, 3});
  EXPECT_EQ(1, array.number_of_axes());
  EXPECT_EQ(3, array.count());
  EXPECT_EQ(3, array.shape(0));

  EXPECT_EQ(1, array.data(0));
  EXPECT_EQ(2, array.data(1));
  EXPECT_EQ(3, array.data(2));
}

TEST(Array, Initialize) {
  auto array = Array<float>({1, 2, 3});
  EXPECT_EQ(3, array.number_of_axes());
  EXPECT_EQ(6, array.count());
  EXPECT_EQ(1, array.shape(0));
  EXPECT_EQ(2, array.shape(1));
  EXPECT_EQ(3, array.shape(2));

  for (auto j = 0; j < 2; ++j) {
    for (auto k = 0; k < 3; ++k) {
      EXPECT_EQ(0, array.at({0, j, k}));
    }
  }

  EXPECT_DEATH(array.at({1, 0, 0}), ".*");
  EXPECT_DEATH(array.at({0, 2, 0}), ".*");
  EXPECT_DEATH(array.at({0, 0, 3}), ".*");

  array.Reshape({2, 3, 4});
  EXPECT_EQ(24, array.count());
  EXPECT_EQ(2, array.shape(0));
  EXPECT_EQ(3, array.shape(1));
  EXPECT_EQ(4, array.shape(2));

  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 3; ++j) {
      for (auto k = 0; k < 4; ++k) {
        EXPECT_EQ(0, array.at({i, j, k}));
      }
    }
  }

  EXPECT_DEATH(array.at({2, 0, 0}), ".*");
  EXPECT_DEATH(array.at({0, 3, 0}), ".*");
  EXPECT_DEATH(array.at({0, 0, 4}), ".*");

  EXPECT_DEATH(array.Reshape({4096, 4096, 4096}), ".*");
}

TEST(Array, InitializeValue) {
  auto array = Array<float>({1, 2, 3}, {0, 1, 2, 3, 4, 5});

  for (auto j = 0; j < 2; ++j) {
    for (auto k = 0; k < 3; ++k) {
      EXPECT_EQ(3 * j + k, array.at({0, j, k}));
    }
  }
}

TEST(Array, Assign) {
  auto array = Array<float>({1, 2, 3});

  auto value = 0;
  for (auto j = 0; j < 2; ++j) {
    for (auto k = 0; k < 3; ++k) {
      array.set({0, j, k}, value++);
    }
  }

  EXPECT_EQ(0, array.at({0, 0, 0}));
  EXPECT_EQ(1, array.at({0, 0, 1}));
  EXPECT_EQ(2, array.at({0, 0, 2}));
  EXPECT_EQ(3, array.at({0, 1, 0}));
  EXPECT_EQ(4, array.at({0, 1, 1}));
  EXPECT_EQ(5, array.at({0, 1, 2}));

  EXPECT_EQ(0, array.data(0));
  EXPECT_EQ(1, array.data(1));
  EXPECT_EQ(2, array.data(2));
  EXPECT_EQ(3, array.data(3));
  EXPECT_EQ(4, array.data(4));
  EXPECT_EQ(5, array.data(5));
}
