#include <gtest/gtest.h>

#include "tensor.hpp"

using sacred::Tensor;

TEST(TestTensor, At) {
  auto tensor = Tensor<float>();
  auto tensor2 = Tensor<float>(3);
  Tensor<float> tensor3 = 4.0f;
  auto tensor4 = Tensor<float>({2, 2});

  EXPECT_EQ(std::vector<int>({1}), tensor2.shape());
  EXPECT_EQ(3.0f, static_cast<int>(tensor2));

  EXPECT_EQ(std::vector<int>({1}), tensor3.shape());
  EXPECT_EQ(4.0f, static_cast<int>(tensor3));

  tensor4 = 5.0f;
  EXPECT_EQ(std::vector<int>({2, 2}), tensor4.shape());
  EXPECT_EQ(5.0f, tensor4.at({0, 0}));
  EXPECT_EQ(5.0f, tensor4.at({0, 1}));
  EXPECT_EQ(5.0f, tensor4.at({1, 0}));
  EXPECT_EQ(5.0f, tensor4.at({1, 1}));
}

TEST(Tensor, Initialize1D) {
  auto tensor = Tensor<float>({3}, {
    1, 2, 3
  });

  EXPECT_EQ(1, tensor.number_of_axes());
  EXPECT_EQ(3, tensor.size());
  EXPECT_EQ(3, tensor.shape().at(0));

  EXPECT_EQ(1, tensor.at({0}));
  EXPECT_EQ(2, tensor.at({1}));
  EXPECT_EQ(3, tensor.at({2}));

  EXPECT_DEATH(tensor.at({-1}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({3}), "indices\\.at\\(i\\).* <");

  EXPECT_DEATH(tensor.at({0, 0}), "indices\\.size\\(\\)");

  // EXPECT_EQ(Tensor<float>({3}, {1, 2, 3}), tensor.slice(_));
}

TEST(Tensor, Initialize2D) {
  auto tensor = Tensor<float>({2, 3}, {
    1, 2, 3,
    4, 5, 6,
  });

  EXPECT_EQ(2, tensor.number_of_axes());
  EXPECT_EQ(6, tensor.size());
  EXPECT_EQ(2, tensor.shape().at(0));
  EXPECT_EQ(3, tensor.shape().at(1));

  EXPECT_EQ(1, tensor.at({0, 0}));
  EXPECT_EQ(2, tensor.at({0, 1}));
  EXPECT_EQ(3, tensor.at({0, 2}));
  EXPECT_EQ(4, tensor.at({1, 0}));
  EXPECT_EQ(5, tensor.at({1, 1}));
  EXPECT_EQ(6, tensor.at({1, 2}));

  EXPECT_DEATH(tensor.at({0, -1}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({-1, 0}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 3}), "indices\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({2, 0}), "indices\\.at\\(i\\).* <");

  // EXPECT_DEATH(tensor.at({0}), "indices\\.size\\(\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0}), "indices\\.size\\(\\)");

  // EXPECT_EQ(Tensor<float>({2}, {1, 4}), tensor.slice(_, 0));
  // EXPECT_EQ(Tensor<float>({2}, {2, 5}), tensor.slice(_, 1));
  // EXPECT_EQ(Tensor<float>({2}, {3, 6}), tensor.slice(_, 2));
  //
  // EXPECT_EQ(Tensor<float>({3}, {1, 2, 3}), tensor.slice(0, _));
  // EXPECT_EQ(Tensor<float>({3}, {4, 5, 6}), tensor.slice(1, _));
}

TEST(Tensor, Initialize3D) {
  auto tensor = Tensor<float>({2, 2, 2}, {
    1, 2,
    3, 4,

    5, 6,
    7, 8,
  });

  EXPECT_EQ(3, tensor.number_of_axes());
  EXPECT_EQ(8, tensor.size());
  EXPECT_EQ(2, tensor.shape().at(0));
  EXPECT_EQ(2, tensor.shape().at(1));
  EXPECT_EQ(2, tensor.shape().at(2));

  EXPECT_EQ(1, tensor.at({0, 0, 0}));
  EXPECT_EQ(2, tensor.at({0, 0, 1}));
  EXPECT_EQ(3, tensor.at({0, 1, 0}));
  EXPECT_EQ(4, tensor.at({0, 1, 1}));
  EXPECT_EQ(5, tensor.at({1, 0, 0}));
  EXPECT_EQ(6, tensor.at({1, 0, 1}));
  EXPECT_EQ(7, tensor.at({1, 1, 0}));
  EXPECT_EQ(8, tensor.at({1, 1, 1}));

  EXPECT_DEATH(tensor.at({0, 0, -1}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, -1, 0}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({-1, 0, 0}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 0, 2}), "indices\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 2, 0}), "indices\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({2, 0, 0}), "indices\\.at\\(i\\).* <");

  // EXPECT_DEATH(tensor.at(0, 0), "indices\\.size\\(\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0, 0}), "indices\\.size\\(\\)");

  // EXPECT_EQ(Tensor<float>({2}, {1, 5}), tensor.slice(_, 0, 0));
  // EXPECT_EQ(Tensor<float>({2}, {2, 6}), tensor.slice(_, 0, 1));
  // EXPECT_EQ(Tensor<float>({2}, {3, 7}), tensor.slice(_, 1, 0));
  // EXPECT_EQ(Tensor<float>({2}, {4, 8}), tensor.slice(_, 1, 1));
  //
  // EXPECT_EQ(Tensor<float>({2}, {1, 3}), tensor.slice(0, _, 0));
  // EXPECT_EQ(Tensor<float>({2}, {2, 4}), tensor.slice(0, _, 1));
  // EXPECT_EQ(Tensor<float>({2}, {5, 7}), tensor.slice(1, _, 0));
  // EXPECT_EQ(Tensor<float>({2}, {6, 8}), tensor.slice(1, _, 1));
  //
  // EXPECT_EQ(Tensor<float>({2}, {1, 2}), tensor.slice(0, 0, _));
  // EXPECT_EQ(Tensor<float>({2}, {3, 4}), tensor.slice(0, 1, _));
  // EXPECT_EQ(Tensor<float>({2}, {5, 6}), tensor.slice(1, 0, _));
  // EXPECT_EQ(Tensor<float>({2}, {7, 8}), tensor.slice(1, 1, _));
}

TEST(Tensor, Initialize4D) {
  auto tensor = Tensor<float>({2, 2, 2, 2}, {
    //{
      1, 2,
      3, 4,

      5, 6,
      7, 8,
    //}, {
      9, 10,
      11, 12,

      13, 14,
      15, 16,
    //}
  });

  EXPECT_EQ(4, tensor.number_of_axes());
  EXPECT_EQ(16, tensor.size());
  EXPECT_EQ(2, tensor.shape().at(0));
  EXPECT_EQ(2, tensor.shape().at(1));
  EXPECT_EQ(2, tensor.shape().at(2));
  EXPECT_EQ(2, tensor.shape().at(2));

  EXPECT_EQ(1, tensor.at({0, 0, 0, 0}));
  EXPECT_EQ(2, tensor.at({0, 0, 0, 1}));
  EXPECT_EQ(3, tensor.at({0, 0, 1, 0}));
  EXPECT_EQ(4, tensor.at({0, 0, 1, 1}));
  EXPECT_EQ(5, tensor.at({0, 1, 0, 0}));
  EXPECT_EQ(6, tensor.at({0, 1, 0, 1}));
  EXPECT_EQ(7, tensor.at({0, 1, 1, 0}));
  EXPECT_EQ(8, tensor.at({0, 1, 1, 1}));
  EXPECT_EQ(9, tensor.at({1, 0, 0, 0}));
  EXPECT_EQ(10, tensor.at({1, 0, 0, 1}));
  EXPECT_EQ(11, tensor.at({1, 0, 1, 0}));
  EXPECT_EQ(12, tensor.at({1, 0, 1, 1}));
  EXPECT_EQ(13, tensor.at({1, 1, 0, 0}));
  EXPECT_EQ(14, tensor.at({1, 1, 0, 1}));
  EXPECT_EQ(15, tensor.at({1, 1, 1, 0}));
  EXPECT_EQ(16, tensor.at({1, 1, 1, 1}));

  EXPECT_DEATH(tensor.at({0, 0, 0, -1}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 0, -1, 0}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, -1, 0, 0}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({-1, 0, 0, 0}), "0 <= indices\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0, 2}), "indices\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 0, 2, 0}), "indices\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 2, 0, 0}), "indices\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({2, 0, 0, 0}), "indices\\.at\\(i\\).* <");

  // EXPECT_DEATH(tensor.at(0, 0, 0), "indices\\.size\\(\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0, 0, 0}), "indices\\.size\\(\\)");

  // EXPECT_EQ(Tensor<float>({2}, {1, 9}), tensor.slice(_, 0, 0, 0));
  // EXPECT_EQ(Tensor<float>({2}, {2, 10}), tensor.slice(_, 0, 0, 1));
  // EXPECT_EQ(Tensor<float>({2}, {3, 11}), tensor.slice(_, 0, 1, 0));
  // EXPECT_EQ(Tensor<float>({2}, {4, 12}), tensor.slice(_, 0, 1, 1));
  // EXPECT_EQ(Tensor<float>({2}, {5, 13}), tensor.slice(_, 1, 0, 0));
  // EXPECT_EQ(Tensor<float>({2}, {6, 14}), tensor.slice(_, 1, 0, 1));
  // EXPECT_EQ(Tensor<float>({2}, {7, 15}), tensor.slice(_, 1, 1, 0));
  // EXPECT_EQ(Tensor<float>({2}, {8, 16}), tensor.slice(_, 1, 1, 1));
  //
  // EXPECT_EQ(Tensor<float>({2}, {1, 5}), tensor.slice(0, _, 0, 0));
  // EXPECT_EQ(Tensor<float>({2}, {2, 6}), tensor.slice(0, _, 0, 1));
  // EXPECT_EQ(Tensor<float>({2}, {3, 7}), tensor.slice(0, _, 1, 0));
  // EXPECT_EQ(Tensor<float>({2}, {4, 8}), tensor.slice(0, _, 1, 1));
  // EXPECT_EQ(Tensor<float>({2}, {9, 13}), tensor.slice(1, _, 0, 0));
  // EXPECT_EQ(Tensor<float>({2}, {10, 14}), tensor.slice(1, _, 0, 1));
  // EXPECT_EQ(Tensor<float>({2}, {11, 15}), tensor.slice(1, _, 1, 0));
  // EXPECT_EQ(Tensor<float>({2}, {12, 16}), tensor.slice(1, _, 1, 1));
  //
  // EXPECT_EQ(Tensor<float>({2}, {1, 3}), tensor.slice(0, 0, _, 0));
  // EXPECT_EQ(Tensor<float>({2}, {2, 4}), tensor.slice(0, 0, _, 1));
  // EXPECT_EQ(Tensor<float>({2}, {5, 7}), tensor.slice(0, 1, _, 0));
  // EXPECT_EQ(Tensor<float>({2}, {6, 8}), tensor.slice(0, 1, _, 1));
  // EXPECT_EQ(Tensor<float>({2}, {9, 11}), tensor.slice(1, 0, _, 0));
  // EXPECT_EQ(Tensor<float>({2}, {10, 12}), tensor.slice(1, 0, _, 1));
  // EXPECT_EQ(Tensor<float>({2}, {13, 15}), tensor.slice(1, 1, _, 0));
  // EXPECT_EQ(Tensor<float>({2}, {14, 16}), tensor.slice(1, 1, _, 1));
  //
  // EXPECT_EQ(Tensor<float>({2}, {1, 2}), tensor.slice(0, 0, 0, _));
  // EXPECT_EQ(Tensor<float>({2}, {3, 4}), tensor.slice(0, 0, 1, _));
  // EXPECT_EQ(Tensor<float>({2}, {5, 6}), tensor.slice(0, 1, 0, _));
  // EXPECT_EQ(Tensor<float>({2}, {7, 8}), tensor.slice(0, 1, 1, _));
  // EXPECT_EQ(Tensor<float>({2}, {9, 10}), tensor.slice(1, 0, 0, _));
  // EXPECT_EQ(Tensor<float>({2}, {11, 12}), tensor.slice(1, 0, 1, _));
  // EXPECT_EQ(Tensor<float>({2}, {13, 14}), tensor.slice(1, 1, 0, _));
  // EXPECT_EQ(Tensor<float>({2}, {15, 16}), tensor.slice(1, 1, 1, _));
}
