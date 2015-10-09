#include <gtest/gtest.h>

#include "tensor.hpp"

using sacred::Tensor;

TEST(Tensor, Initialize1D) {
  auto tensor = Tensor<float>({3}, {1, 2, 3});

  EXPECT_EQ(1, tensor.number_of_axes());
  EXPECT_EQ(3, tensor.size());
  EXPECT_EQ(3, tensor.shape(0));

  EXPECT_DEATH(tensor.shape(-1), "shape_\\.size\\(\\)");
  EXPECT_DEATH(tensor.shape(1), "shape_\\.size\\(\\)");

  EXPECT_EQ(1, tensor.at(0));
  EXPECT_EQ(2, tensor.at(1));
  EXPECT_EQ(3, tensor.at(2));

  EXPECT_EQ(1, tensor.at({0}));
  EXPECT_EQ(2, tensor.at({1}));
  EXPECT_EQ(3, tensor.at({2}));

  EXPECT_DEATH(tensor.at({}), "indices\\.size\\(\\)");
  EXPECT_DEATH(tensor.at({0, 0}), "indices\\.size\\(\\)");
}

TEST(Tensor, Initialize2D) {
  auto tensor = Tensor<float>({2, 3}, {1, 2, 3, 4, 5, 6});

  EXPECT_EQ(2, tensor.number_of_axes());
  EXPECT_EQ(6, tensor.size());
  EXPECT_EQ(2, tensor.shape(0));
  EXPECT_EQ(3, tensor.shape(1));

  EXPECT_DEATH(tensor.shape(-1), "shape_\\.size\\(\\)");
  EXPECT_DEATH(tensor.shape(2), "shape_\\.size\\(\\)");

  EXPECT_EQ(1, tensor.at(0));
  EXPECT_EQ(2, tensor.at(1));
  EXPECT_EQ(3, tensor.at(2));
  EXPECT_EQ(4, tensor.at(3));
  EXPECT_EQ(5, tensor.at(4));
  EXPECT_EQ(6, tensor.at(5));

  EXPECT_EQ(1, tensor.at({0, 0}));
  EXPECT_EQ(2, tensor.at({0, 1}));
  EXPECT_EQ(3, tensor.at({0, 2}));
  EXPECT_EQ(4, tensor.at({1, 0}));
  EXPECT_EQ(5, tensor.at({1, 1}));
  EXPECT_EQ(6, tensor.at({1, 2}));

  EXPECT_DEATH(tensor.at({0}), "indices\\.size\\(\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0}), "indices\\.size\\(\\)");
}

TEST(Tensor, Initialize3D) {
  auto tensor = Tensor<float>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

  EXPECT_EQ(3, tensor.number_of_axes());
  EXPECT_EQ(8, tensor.size());
  EXPECT_EQ(2, tensor.shape(0));
  EXPECT_EQ(2, tensor.shape(1));
  EXPECT_EQ(2, tensor.shape(2));

  EXPECT_DEATH(tensor.shape(-1), "shape_\\.size\\(\\)");
  EXPECT_DEATH(tensor.shape(3), "shape_\\.size\\(\\)");

  EXPECT_EQ(1, tensor.at(0));
  EXPECT_EQ(2, tensor.at(1));
  EXPECT_EQ(3, tensor.at(2));
  EXPECT_EQ(4, tensor.at(3));
  EXPECT_EQ(5, tensor.at(4));
  EXPECT_EQ(6, tensor.at(5));
  EXPECT_EQ(7, tensor.at(6));
  EXPECT_EQ(8, tensor.at(7));

  EXPECT_EQ(1, tensor.at({0, 0, 0}));
  EXPECT_EQ(2, tensor.at({0, 0, 1}));
  EXPECT_EQ(3, tensor.at({0, 1, 0}));
  EXPECT_EQ(4, tensor.at({0, 1, 1}));
  EXPECT_EQ(5, tensor.at({1, 0, 0}));
  EXPECT_EQ(6, tensor.at({1, 0, 1}));
  EXPECT_EQ(7, tensor.at({1, 1, 0}));
  EXPECT_EQ(8, tensor.at({1, 1, 1}));

  EXPECT_DEATH(tensor.at({0, 0}), "indices\\.size\\(\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0, 0}), "indices\\.size\\(\\)");
}
