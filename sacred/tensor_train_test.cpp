#include <gtest/gtest.h>

#include "tensor_train.hpp"

namespace sacred {

TEST(TensorTrain, Initialize) {
  auto tensor_train = TensorTrain<>();

  EXPECT_EQ(1, tensor_train.at({0, 0, 0}));
  EXPECT_EQ(2, tensor_train.at({0, 0, 1}));
  EXPECT_EQ(3, tensor_train.at({0, 0, 2}));
  EXPECT_EQ(2, tensor_train.at({0, 1, 0}));
  EXPECT_EQ(4, tensor_train.at({0, 1, 1}));
  EXPECT_EQ(6, tensor_train.at({0, 1, 2}));
  EXPECT_EQ(3, tensor_train.at({0, 2, 0}));
  EXPECT_EQ(6, tensor_train.at({0, 2, 1}));
  EXPECT_EQ(9, tensor_train.at({0, 2, 2}));
  EXPECT_EQ(2, tensor_train.at({1, 0, 0}));
  EXPECT_EQ(4, tensor_train.at({1, 0, 1}));
  EXPECT_EQ(6, tensor_train.at({1, 0, 2}));
  EXPECT_EQ(4, tensor_train.at({1, 1, 0}));
  EXPECT_EQ(8, tensor_train.at({1, 1, 1}));
  EXPECT_EQ(12, tensor_train.at({1, 1, 2}));
  EXPECT_EQ(6, tensor_train.at({1, 2, 0}));
  EXPECT_EQ(12, tensor_train.at({1, 2, 1}));
  EXPECT_EQ(18, tensor_train.at({1, 2, 2}));
  EXPECT_EQ(3, tensor_train.at({2, 0, 0}));
  EXPECT_EQ(6, tensor_train.at({2, 0, 1}));
  EXPECT_EQ(9, tensor_train.at({2, 0, 2}));
  EXPECT_EQ(6, tensor_train.at({2, 1, 0}));
  EXPECT_EQ(12, tensor_train.at({2, 1, 1}));
  EXPECT_EQ(18, tensor_train.at({2, 1, 2}));
  EXPECT_EQ(9, tensor_train.at({2, 2, 0}));
  EXPECT_EQ(18, tensor_train.at({2, 2, 1}));
  EXPECT_EQ(27, tensor_train.at({2, 2, 2}));
}

}  // namespace sacred
