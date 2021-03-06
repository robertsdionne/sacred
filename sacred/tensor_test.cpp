#include <gtest/gtest.h>

#include "indexing/mirrored_index.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(TestTensor, At) {
  auto tensor = Tensor<>();
  auto tensor2 = Tensor<>(3);
  Tensor<> tensor3 = 4.0f;
  auto tensor4 = Tensor<>({2, 2});

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

  auto tensor5 = Tensor<>({3, 3}, {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
  });

  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-3, -3}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({-3, -2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-3, -1}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({-3, 0}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-3, 1}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({-3, 2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-3, 3}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({-3, 4}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-3, 5}));

  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({-2, -3}));
  EXPECT_EQ(9, tensor5.at<indexing::MirroredIndex<>>({-2, -2}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({-2, -1}));
  EXPECT_EQ(7, tensor5.at<indexing::MirroredIndex<>>({-2, 0}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({-2, 1}));
  EXPECT_EQ(9, tensor5.at<indexing::MirroredIndex<>>({-2, 2}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({-2, 3}));
  EXPECT_EQ(7, tensor5.at<indexing::MirroredIndex<>>({-2, 4}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({-2, 5}));

  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-1, -3}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({-1, -2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-1, -1}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({-1, 0}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-1, 1}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({-1, 2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-1, 3}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({-1, 4}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({-1, 5}));

  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({0, -3}));
  EXPECT_EQ(3, tensor5.at<indexing::MirroredIndex<>>({0, -2}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({0, -1}));
  EXPECT_EQ(1, tensor5.at<indexing::MirroredIndex<>>({0, 0}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({0, 1}));
  EXPECT_EQ(3, tensor5.at<indexing::MirroredIndex<>>({0, 2}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({0, 3}));
  EXPECT_EQ(1, tensor5.at<indexing::MirroredIndex<>>({0, 4}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({0, 5}));

  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({1, -3}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({1, -2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({1, -1}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({1, 0}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({1, 1}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({1, 2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({1, 3}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({1, 4}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({1, 5}));

  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({2, -3}));
  EXPECT_EQ(9, tensor5.at<indexing::MirroredIndex<>>({2, -2}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({2, -1}));
  EXPECT_EQ(7, tensor5.at<indexing::MirroredIndex<>>({2, 0}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({2, 1}));
  EXPECT_EQ(9, tensor5.at<indexing::MirroredIndex<>>({2, 2}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({2, 3}));
  EXPECT_EQ(7, tensor5.at<indexing::MirroredIndex<>>({2, 4}));
  EXPECT_EQ(8, tensor5.at<indexing::MirroredIndex<>>({2, 5}));

  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({3, -3}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({3, -2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({3, -1}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({3, 0}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({3, 1}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({3, 2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({3, 3}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({3, 4}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({3, 5}));

  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({4, -3}));
  EXPECT_EQ(3, tensor5.at<indexing::MirroredIndex<>>({4, -2}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({4, -1}));
  EXPECT_EQ(1, tensor5.at<indexing::MirroredIndex<>>({4, 0}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({4, 1}));
  EXPECT_EQ(3, tensor5.at<indexing::MirroredIndex<>>({4, 2}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({4, 3}));
  EXPECT_EQ(1, tensor5.at<indexing::MirroredIndex<>>({4, 4}));
  EXPECT_EQ(2, tensor5.at<indexing::MirroredIndex<>>({4, 5}));

  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({5, -3}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({5, -2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({5, -1}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({5, 0}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({5, 1}));
  EXPECT_EQ(6, tensor5.at<indexing::MirroredIndex<>>({5, 2}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({5, 3}));
  EXPECT_EQ(4, tensor5.at<indexing::MirroredIndex<>>({5, 4}));
  EXPECT_EQ(5, tensor5.at<indexing::MirroredIndex<>>({5, 5}));
}

TEST(Tensor, Initialize1D) {
  auto tensor = Tensor<float, long>({3}, {
    1, 2, 3
  });

  EXPECT_EQ(1, tensor.number_of_axes());
  EXPECT_EQ(3, tensor.size());
  EXPECT_EQ(3, tensor.shape().at(0));

  EXPECT_EQ(1, tensor.at({0}));
  EXPECT_EQ(2, tensor.at({1}));
  EXPECT_EQ(3, tensor.at({2}));

  EXPECT_DEATH(tensor.at({-1}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({3}), "index\\.at\\(i\\).* <");

  EXPECT_EQ(3, tensor[{-1}]);
  EXPECT_EQ(1, tensor[{0}]);
  EXPECT_EQ(2, tensor[{1}]);
  EXPECT_EQ(3, tensor[{2}]);
  EXPECT_EQ(1, tensor[{3}]);

  EXPECT_DEATH(tensor.at({0, 0}), "index\\.size\\(\\)");
}

TEST(Tensor, Initialize2D) {
  auto tensor = Tensor<>({2, 3}, {
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

  EXPECT_DEATH(tensor.at({-1, 0}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, -1}), "0 <= index\\.at\\(i\\)");

  EXPECT_DEATH(tensor.at({2, 0}), "index\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 3}), "index\\.at\\(i\\).* <");

  EXPECT_EQ(6, (tensor[{-1, -1}]));
  EXPECT_EQ(4, (tensor[{-1, 0}]));
  EXPECT_EQ(5, (tensor[{-1, 1}]));
  EXPECT_EQ(6, (tensor[{-1, 2}]));
  EXPECT_EQ(4, (tensor[{-1, 3}]));

  EXPECT_EQ(3, (tensor[{0, -1}]));
  EXPECT_EQ(1, (tensor[{0, 0}]));
  EXPECT_EQ(2, (tensor[{0, 1}]));
  EXPECT_EQ(3, (tensor[{0, 2}]));
  EXPECT_EQ(1, (tensor[{0, 3}]));

  EXPECT_EQ(6, (tensor[{1, -1}]));
  EXPECT_EQ(4, (tensor[{1, 0}]));
  EXPECT_EQ(5, (tensor[{1, 1}]));
  EXPECT_EQ(6, (tensor[{1, 2}]));
  EXPECT_EQ(4, (tensor[{1, 3}]));

  EXPECT_EQ(3, (tensor[{2, -1}]));
  EXPECT_EQ(1, (tensor[{2, 0}]));
  EXPECT_EQ(2, (tensor[{2, 1}]));
  EXPECT_EQ(3, (tensor[{2, 2}]));
  EXPECT_EQ(1, (tensor[{2, 3}]));

  EXPECT_DEATH(tensor.at({0, 0, 0}), "index\\.size\\(\\)");
}

TEST(Tensor, Initialize3D) {
  auto tensor = Tensor<>({2, 2, 2}, {
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

  EXPECT_DEATH(tensor.at({-1, 0, 0}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, -1, 0}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 0, -1}), "0 <= index\\.at\\(i\\)");

  EXPECT_DEATH(tensor.at({2, 0, 0}), "index\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 2, 0}), "index\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 0, 2}), "index\\.at\\(i\\).* <");

  EXPECT_DEATH(tensor.at({0, 0, 0, 0}), "index\\.size\\(\\)");

  EXPECT_EQ(8, (tensor[{-1, -1, -1}]));
  EXPECT_EQ(7, (tensor[{-1, -1, 0}]));
  EXPECT_EQ(8, (tensor[{-1, -1, 1}]));
  EXPECT_EQ(7, (tensor[{-1, -1, 2}]));

  EXPECT_EQ(6, (tensor[{-1, 0, -1}]));
  EXPECT_EQ(5, (tensor[{-1, 0, 0}]));
  EXPECT_EQ(6, (tensor[{-1, 0, 1}]));
  EXPECT_EQ(5, (tensor[{-1, 0, 2}]));

  EXPECT_EQ(8, (tensor[{-1, 1, -1}]));
  EXPECT_EQ(7, (tensor[{-1, 1, 0}]));
  EXPECT_EQ(8, (tensor[{-1, 1, 1}]));
  EXPECT_EQ(7, (tensor[{-1, 1, 2}]));

  EXPECT_EQ(6, (tensor[{-1, 2, -1}]));
  EXPECT_EQ(5, (tensor[{-1, 2, 0}]));
  EXPECT_EQ(6, (tensor[{-1, 2, 1}]));
  EXPECT_EQ(5, (tensor[{-1, 2, 2}]));

  EXPECT_EQ(4, (tensor[{0, -1, -1}]));
  EXPECT_EQ(3, (tensor[{0, -1, 0}]));
  EXPECT_EQ(4, (tensor[{0, -1, 1}]));
  EXPECT_EQ(3, (tensor[{0, -1, 2}]));

  EXPECT_EQ(2, (tensor[{0, 0, -1}]));
  EXPECT_EQ(1, (tensor[{0, 0, 0}]));
  EXPECT_EQ(2, (tensor[{0, 0, 1}]));
  EXPECT_EQ(1, (tensor[{0, 0, 2}]));

  EXPECT_EQ(4, (tensor[{0, 1, -1}]));
  EXPECT_EQ(3, (tensor[{0, 1, 0}]));
  EXPECT_EQ(4, (tensor[{0, 1, 1}]));
  EXPECT_EQ(3, (tensor[{0, 1, 2}]));

  EXPECT_EQ(2, (tensor[{0, 2, -1}]));
  EXPECT_EQ(1, (tensor[{0, 2, 0}]));
  EXPECT_EQ(2, (tensor[{0, 2, 1}]));
  EXPECT_EQ(1, (tensor[{0, 2, 2}]));

  EXPECT_EQ(8, (tensor[{1, -1, -1}]));
  EXPECT_EQ(7, (tensor[{1, -1, 0}]));
  EXPECT_EQ(8, (tensor[{1, -1, 1}]));
  EXPECT_EQ(7, (tensor[{1, -1, 2}]));

  EXPECT_EQ(6, (tensor[{1, 0, -11}]));
  EXPECT_EQ(5, (tensor[{1, 0, 0}]));
  EXPECT_EQ(6, (tensor[{1, 0, 1}]));
  EXPECT_EQ(5, (tensor[{1, 0, 2}]));

  EXPECT_EQ(8, (tensor[{1, 1, -1}]));
  EXPECT_EQ(7, (tensor[{1, 1, 0}]));
  EXPECT_EQ(8, (tensor[{1, 1, 1}]));
  EXPECT_EQ(7, (tensor[{1, 1, 2}]));

  EXPECT_EQ(6, (tensor[{1, 2, -1}]));
  EXPECT_EQ(5, (tensor[{1, 2, 0}]));
  EXPECT_EQ(6, (tensor[{1, 2, 1}]));
  EXPECT_EQ(5, (tensor[{1, 2, 2}]));

  EXPECT_EQ(4, (tensor[{2, -1, -1}]));
  EXPECT_EQ(3, (tensor[{2, -1, 0}]));
  EXPECT_EQ(4, (tensor[{2, -1, 1}]));
  EXPECT_EQ(3, (tensor[{2, -1, 2}]));

  EXPECT_EQ(2, (tensor[{2, 0, -11}]));
  EXPECT_EQ(1, (tensor[{2, 0, 0}]));
  EXPECT_EQ(2, (tensor[{2, 0, 1}]));
  EXPECT_EQ(1, (tensor[{2, 0, 2}]));

  EXPECT_EQ(4, (tensor[{2, 1, -1}]));
  EXPECT_EQ(3, (tensor[{2, 1, 0}]));
  EXPECT_EQ(4, (tensor[{2, 1, 1}]));
  EXPECT_EQ(3, (tensor[{2, 1, 2}]));

  EXPECT_EQ(2, (tensor[{2, 2, -1}]));
  EXPECT_EQ(1, (tensor[{2, 2, 0}]));
  EXPECT_EQ(2, (tensor[{2, 2, 1}]));
  EXPECT_EQ(1, (tensor[{2, 2, 2}]));
}

TEST(Tensor, Initialize4D) {
  auto tensor = Tensor<>({2, 2, 2, 2}, {
      1, 2,
      3, 4,

      5, 6,
      7, 8,

      9, 10,
      11, 12,

      13, 14,
      15, 16,
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

  EXPECT_DEATH(tensor.at({-1, 0, 0, 0}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, -1, 0, 0}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 0, -1, 0}), "0 <= index\\.at\\(i\\)");
  EXPECT_DEATH(tensor.at({0, 0, 0, -1}), "0 <= index\\.at\\(i\\)");

  EXPECT_DEATH(tensor.at({2, 0, 0, 0}), "index\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 2, 0, 0}), "index\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 0, 2, 0}), "index\\.at\\(i\\).* <");
  EXPECT_DEATH(tensor.at({0, 0, 0, 2}), "index\\.at\\(i\\).* <");

  EXPECT_DEATH(tensor.at({0, 0, 0, 0, 0}), "index\\.size\\(\\)");

  EXPECT_EQ(16, (tensor[{-1, -1, -1, -1}]));
  EXPECT_EQ(15, (tensor[{-1, -1, -1, 0}]));
  EXPECT_EQ(16, (tensor[{-1, -1, -1, 1}]));
  EXPECT_EQ(15, (tensor[{-1, -1, -1, 2}]));

  EXPECT_EQ(14, (tensor[{-1, -1, 0, -1}]));
  EXPECT_EQ(13, (tensor[{-1, -1, 0, 0}]));
  EXPECT_EQ(14, (tensor[{-1, -1, 0, 1}]));
  EXPECT_EQ(13, (tensor[{-1, -1, 0, 2}]));

  EXPECT_EQ(16, (tensor[{-1, -1, 1, -1}]));
  EXPECT_EQ(15, (tensor[{-1, -1, 1, 0}]));
  EXPECT_EQ(16, (tensor[{-1, -1, 1, 1}]));
  EXPECT_EQ(15, (tensor[{-1, -1, 1, 2}]));

  EXPECT_EQ(14, (tensor[{-1, -1, 2, -1}]));
  EXPECT_EQ(13, (tensor[{-1, -1, 2, 0}]));
  EXPECT_EQ(14, (tensor[{-1, -1, 2, 1}]));
  EXPECT_EQ(13, (tensor[{-1, -1, 2, 2}]));

  EXPECT_EQ(12, (tensor[{-1, 0, -1, -1}]));
  EXPECT_EQ(11, (tensor[{-1, 0, -1, 0}]));
  EXPECT_EQ(12, (tensor[{-1, 0, -1, 1}]));
  EXPECT_EQ(11, (tensor[{-1, 0, -1, 2}]));

  EXPECT_EQ(10, (tensor[{-1, 0, 0, -1}]));
  EXPECT_EQ(9, (tensor[{-1, 0, 0, 0}]));
  EXPECT_EQ(10, (tensor[{-1, 0, 0, 1}]));
  EXPECT_EQ(9, (tensor[{-1, 0, 0, 2}]));

  EXPECT_EQ(12, (tensor[{-1, 0, 1, -1}]));
  EXPECT_EQ(11, (tensor[{-1, 0, 1, 0}]));
  EXPECT_EQ(12, (tensor[{-1, 0, 1, 1}]));
  EXPECT_EQ(11, (tensor[{-1, 0, 1, 2}]));

  EXPECT_EQ(10, (tensor[{-1, 0, 2, -1}]));
  EXPECT_EQ(9, (tensor[{-1, 0, 2, 0}]));
  EXPECT_EQ(10, (tensor[{-1, 0, 2, 1}]));
  EXPECT_EQ(9, (tensor[{-1, 0, 2, 2}]));

  EXPECT_EQ(16, (tensor[{-1, 1, -1, -1}]));
  EXPECT_EQ(15, (tensor[{-1, 1, -1, 0}]));
  EXPECT_EQ(16, (tensor[{-1, 1, -1, 1}]));
  EXPECT_EQ(15, (tensor[{-1, 1, -1, 2}]));

  EXPECT_EQ(14, (tensor[{-1, 1, 0, -1}]));
  EXPECT_EQ(13, (tensor[{-1, 1, 0, 0}]));
  EXPECT_EQ(14, (tensor[{-1, 1, 0, 1}]));
  EXPECT_EQ(13, (tensor[{-1, 1, 0, 2}]));

  EXPECT_EQ(16, (tensor[{-1, 1, 1, -1}]));
  EXPECT_EQ(15, (tensor[{-1, 1, 1, 0}]));
  EXPECT_EQ(16, (tensor[{-1, 1, 1, 1}]));
  EXPECT_EQ(15, (tensor[{-1, 1, 1, 2}]));

  EXPECT_EQ(14, (tensor[{-1, 1, 2, -1}]));
  EXPECT_EQ(13, (tensor[{-1, 1, 2, 0}]));
  EXPECT_EQ(14, (tensor[{-1, 1, 2, 1}]));
  EXPECT_EQ(13, (tensor[{-1, 1, 2, 2}]));

  EXPECT_EQ(12, (tensor[{-1, 2, -1, -1}]));
  EXPECT_EQ(11, (tensor[{-1, 2, -1, 0}]));
  EXPECT_EQ(12, (tensor[{-1, 2, -1, 1}]));
  EXPECT_EQ(11, (tensor[{-1, 2, -1, 2}]));

  EXPECT_EQ(10, (tensor[{-1, 2, 0, -1}]));
  EXPECT_EQ(9, (tensor[{-1, 2, 0, 0}]));
  EXPECT_EQ(10, (tensor[{-1, 2, 0, 1}]));
  EXPECT_EQ(9, (tensor[{-1, 2, 0, 2}]));

  EXPECT_EQ(12, (tensor[{-1, 2, 1, -1}]));
  EXPECT_EQ(11, (tensor[{-1, 2, 1, 0}]));
  EXPECT_EQ(12, (tensor[{-1, 2, 1, 1}]));
  EXPECT_EQ(11, (tensor[{-1, 2, 1, 2}]));

  EXPECT_EQ(10, (tensor[{-1, 2, 2, -1}]));
  EXPECT_EQ(9, (tensor[{-1, 2, 2, 0}]));
  EXPECT_EQ(10, (tensor[{-1, 2, 2, 1}]));
  EXPECT_EQ(9, (tensor[{-1, 2, 2, 2}]));

  EXPECT_EQ(8, (tensor[{0, -1, -1, -1}]));
  EXPECT_EQ(7, (tensor[{0, -1, -1, 0}]));
  EXPECT_EQ(8, (tensor[{0, -1, -1, 1}]));
  EXPECT_EQ(7, (tensor[{0, -1, -1, 2}]));

  EXPECT_EQ(6, (tensor[{0, -1, 0, -1}]));
  EXPECT_EQ(5, (tensor[{0, -1, 0, 0}]));
  EXPECT_EQ(6, (tensor[{0, -1, 0, 1}]));
  EXPECT_EQ(5, (tensor[{0, -1, 0, 2}]));

  EXPECT_EQ(8, (tensor[{0, -1, 1, -1}]));
  EXPECT_EQ(7, (tensor[{0, -1, 1, 0}]));
  EXPECT_EQ(8, (tensor[{0, -1, 1, 1}]));
  EXPECT_EQ(7, (tensor[{0, -1, 1, 2}]));

  EXPECT_EQ(6, (tensor[{0, -1, 2, -1}]));
  EXPECT_EQ(5, (tensor[{0, -1, 2, 0}]));
  EXPECT_EQ(6, (tensor[{0, -1, 2, 1}]));
  EXPECT_EQ(5, (tensor[{0, -1, 2, 2}]));

  EXPECT_EQ(4, (tensor[{0, 0, -1, -1}]));
  EXPECT_EQ(3, (tensor[{0, 0, -1, 0}]));
  EXPECT_EQ(4, (tensor[{0, 0, -1, 1}]));
  EXPECT_EQ(3, (tensor[{0, 0, -1, 2}]));

  EXPECT_EQ(2, (tensor[{0, 0, 0, -1}]));
  EXPECT_EQ(1, (tensor[{0, 0, 0, 0}]));
  EXPECT_EQ(2, (tensor[{0, 0, 0, 1}]));
  EXPECT_EQ(1, (tensor[{0, 0, 0, 2}]));

  EXPECT_EQ(4, (tensor[{0, 0, 1, -1}]));
  EXPECT_EQ(3, (tensor[{0, 0, 1, 0}]));
  EXPECT_EQ(4, (tensor[{0, 0, 1, 1}]));
  EXPECT_EQ(3, (tensor[{0, 0, 1, 2}]));

  EXPECT_EQ(2, (tensor[{0, 0, 2, -1}]));
  EXPECT_EQ(1, (tensor[{0, 0, 2, 0}]));
  EXPECT_EQ(2, (tensor[{0, 0, 2, 1}]));
  EXPECT_EQ(1, (tensor[{0, 0, 2, 2}]));

  EXPECT_EQ(8, (tensor[{0, 1, -1, -1}]));
  EXPECT_EQ(7, (tensor[{0, 1, -1, 0}]));
  EXPECT_EQ(8, (tensor[{0, 1, -1, 1}]));
  EXPECT_EQ(7, (tensor[{0, 1, -1, 2}]));

  EXPECT_EQ(6, (tensor[{0, 1, 0, -1}]));
  EXPECT_EQ(5, (tensor[{0, 1, 0, 0}]));
  EXPECT_EQ(6, (tensor[{0, 1, 0, 1}]));
  EXPECT_EQ(5, (tensor[{0, 1, 0, 2}]));

  EXPECT_EQ(8, (tensor[{0, 1, 1, -1}]));
  EXPECT_EQ(7, (tensor[{0, 1, 1, 0}]));
  EXPECT_EQ(8, (tensor[{0, 1, 1, 1}]));
  EXPECT_EQ(7, (tensor[{0, 1, 1, 2}]));

  EXPECT_EQ(6, (tensor[{0, 1, 2, -1}]));
  EXPECT_EQ(5, (tensor[{0, 1, 2, 0}]));
  EXPECT_EQ(6, (tensor[{0, 1, 2, 1}]));
  EXPECT_EQ(5, (tensor[{0, 1, 2, 2}]));

  EXPECT_EQ(4, (tensor[{0, 2, -1, -1}]));
  EXPECT_EQ(3, (tensor[{0, 2, -1, 0}]));
  EXPECT_EQ(4, (tensor[{0, 2, -1, 1}]));
  EXPECT_EQ(3, (tensor[{0, 2, -1, 2}]));

  EXPECT_EQ(2, (tensor[{0, 2, 0, -1}]));
  EXPECT_EQ(1, (tensor[{0, 2, 0, 0}]));
  EXPECT_EQ(2, (tensor[{0, 2, 0, 1}]));
  EXPECT_EQ(1, (tensor[{0, 2, 0, 2}]));

  EXPECT_EQ(4, (tensor[{0, 2, 1, -1}]));
  EXPECT_EQ(3, (tensor[{0, 2, 1, 0}]));
  EXPECT_EQ(4, (tensor[{0, 2, 1, 1}]));
  EXPECT_EQ(3, (tensor[{0, 2, 1, 2}]));

  EXPECT_EQ(2, (tensor[{0, 2, 2, -1}]));
  EXPECT_EQ(1, (tensor[{0, 2, 2, 0}]));
  EXPECT_EQ(2, (tensor[{0, 2, 2, 1}]));
  EXPECT_EQ(1, (tensor[{0, 2, 2, 2}]));

  EXPECT_EQ(16, (tensor[{1, -1, -1, -1}]));
  EXPECT_EQ(15, (tensor[{1, -1, -1, 0}]));
  EXPECT_EQ(16, (tensor[{1, -1, -1, 1}]));
  EXPECT_EQ(15, (tensor[{1, -1, -1, 2}]));

  EXPECT_EQ(14, (tensor[{1, -1, 0, -1}]));
  EXPECT_EQ(13, (tensor[{1, -1, 0, 0}]));
  EXPECT_EQ(14, (tensor[{1, -1, 0, 1}]));
  EXPECT_EQ(13, (tensor[{1, -1, 0, 2}]));

  EXPECT_EQ(16, (tensor[{1, -1, 1, -1}]));
  EXPECT_EQ(15, (tensor[{1, -1, 1, 0}]));
  EXPECT_EQ(16, (tensor[{1, -1, 1, 1}]));
  EXPECT_EQ(15, (tensor[{1, -1, 1, 2}]));

  EXPECT_EQ(14, (tensor[{1, -1, 2, -1}]));
  EXPECT_EQ(13, (tensor[{1, -1, 2, 0}]));
  EXPECT_EQ(14, (tensor[{1, -1, 2, 1}]));
  EXPECT_EQ(13, (tensor[{1, -1, 2, 2}]));

  EXPECT_EQ(12, (tensor[{1, 0, -1, -1}]));
  EXPECT_EQ(11, (tensor[{1, 0, -1, 0}]));
  EXPECT_EQ(12, (tensor[{1, 0, -1, 1}]));
  EXPECT_EQ(11, (tensor[{1, 0, -1, 2}]));

  EXPECT_EQ(10, (tensor[{1, 0, 0, -1}]));
  EXPECT_EQ(9, (tensor[{1, 0, 0, 0}]));
  EXPECT_EQ(10, (tensor[{1, 0, 0, 1}]));
  EXPECT_EQ(9, (tensor[{1, 0, 0, 2}]));

  EXPECT_EQ(12, (tensor[{1, 0, 1, -1}]));
  EXPECT_EQ(11, (tensor[{1, 0, 1, 0}]));
  EXPECT_EQ(12, (tensor[{1, 0, 1, 1}]));
  EXPECT_EQ(11, (tensor[{1, 0, 1, 2}]));

  EXPECT_EQ(10, (tensor[{1, 0, 2, -1}]));
  EXPECT_EQ(9, (tensor[{1, 0, 2, 0}]));
  EXPECT_EQ(10, (tensor[{1, 0, 2, 1}]));
  EXPECT_EQ(9, (tensor[{1, 0, 2, 2}]));

  EXPECT_EQ(16, (tensor[{1, 1, -1, -1}]));
  EXPECT_EQ(15, (tensor[{1, 1, -1, 0}]));
  EXPECT_EQ(16, (tensor[{1, 1, -1, 1}]));
  EXPECT_EQ(15, (tensor[{1, 1, -1, 2}]));

  EXPECT_EQ(14, (tensor[{1, 1, 0, -1}]));
  EXPECT_EQ(13, (tensor[{1, 1, 0, 0}]));
  EXPECT_EQ(14, (tensor[{1, 1, 0, 1}]));
  EXPECT_EQ(13, (tensor[{1, 1, 0, 2}]));

  EXPECT_EQ(16, (tensor[{1, 1, 1, -1}]));
  EXPECT_EQ(15, (tensor[{1, 1, 1, 0}]));
  EXPECT_EQ(16, (tensor[{1, 1, 1, 1}]));
  EXPECT_EQ(15, (tensor[{1, 1, 1, 2}]));

  EXPECT_EQ(14, (tensor[{1, 1, 2, -1}]));
  EXPECT_EQ(13, (tensor[{1, 1, 2, 0}]));
  EXPECT_EQ(14, (tensor[{1, 1, 2, 1}]));
  EXPECT_EQ(13, (tensor[{1, 1, 2, 2}]));

  EXPECT_EQ(12, (tensor[{1, 2, -1, -1}]));
  EXPECT_EQ(11, (tensor[{1, 2, -1, 0}]));
  EXPECT_EQ(12, (tensor[{1, 2, -1, 1}]));
  EXPECT_EQ(11, (tensor[{1, 2, -1, 2}]));

  EXPECT_EQ(10, (tensor[{1, 2, 0, -1}]));
  EXPECT_EQ(9, (tensor[{1, 2, 0, 0}]));
  EXPECT_EQ(10, (tensor[{1, 2, 0, 1}]));
  EXPECT_EQ(9, (tensor[{1, 2, 0, 2}]));

  EXPECT_EQ(12, (tensor[{1, 2, 1, -1}]));
  EXPECT_EQ(11, (tensor[{1, 2, 1, 0}]));
  EXPECT_EQ(12, (tensor[{1, 2, 1, 1}]));
  EXPECT_EQ(11, (tensor[{1, 2, 1, 2}]));

  EXPECT_EQ(10, (tensor[{1, 2, 2, -1}]));
  EXPECT_EQ(9, (tensor[{1, 2, 2, 0}]));
  EXPECT_EQ(10, (tensor[{1, 2, 2, 1}]));
  EXPECT_EQ(9, (tensor[{1, 2, 2, 2}]));

  EXPECT_EQ(8, (tensor[{2, -1, -1, -1}]));
  EXPECT_EQ(7, (tensor[{2, -1, -1, 0}]));
  EXPECT_EQ(8, (tensor[{2, -1, -1, 1}]));
  EXPECT_EQ(7, (tensor[{2, -1, -1, 2}]));

  EXPECT_EQ(6, (tensor[{2, -1, 0, -1}]));
  EXPECT_EQ(5, (tensor[{2, -1, 0, 0}]));
  EXPECT_EQ(6, (tensor[{2, -1, 0, 1}]));
  EXPECT_EQ(5, (tensor[{2, -1, 0, 2}]));

  EXPECT_EQ(8, (tensor[{2, -1, 1, -1}]));
  EXPECT_EQ(7, (tensor[{2, -1, 1, 0}]));
  EXPECT_EQ(8, (tensor[{2, -1, 1, 1}]));
  EXPECT_EQ(7, (tensor[{2, -1, 1, 2}]));

  EXPECT_EQ(6, (tensor[{2, -1, 2, -1}]));
  EXPECT_EQ(5, (tensor[{2, -1, 2, 0}]));
  EXPECT_EQ(6, (tensor[{2, -1, 2, 1}]));
  EXPECT_EQ(5, (tensor[{2, -1, 2, 2}]));

  EXPECT_EQ(4, (tensor[{2, 0, -1, -1}]));
  EXPECT_EQ(3, (tensor[{2, 0, -1, 0}]));
  EXPECT_EQ(4, (tensor[{2, 0, -1, 1}]));
  EXPECT_EQ(3, (tensor[{2, 0, -1, 2}]));

  EXPECT_EQ(2, (tensor[{2, 0, 0, -1}]));
  EXPECT_EQ(1, (tensor[{2, 0, 0, 0}]));
  EXPECT_EQ(2, (tensor[{2, 0, 0, 1}]));
  EXPECT_EQ(1, (tensor[{2, 0, 0, 2}]));

  EXPECT_EQ(4, (tensor[{2, 0, 1, -1}]));
  EXPECT_EQ(3, (tensor[{2, 0, 1, 0}]));
  EXPECT_EQ(4, (tensor[{2, 0, 1, 1}]));
  EXPECT_EQ(3, (tensor[{2, 0, 1, 2}]));

  EXPECT_EQ(2, (tensor[{2, 0, 2, -1}]));
  EXPECT_EQ(1, (tensor[{2, 0, 2, 0}]));
  EXPECT_EQ(2, (tensor[{2, 0, 2, 1}]));
  EXPECT_EQ(1, (tensor[{2, 0, 2, 2}]));

  EXPECT_EQ(8, (tensor[{2, 1, -1, -1}]));
  EXPECT_EQ(7, (tensor[{2, 1, -1, 0}]));
  EXPECT_EQ(8, (tensor[{2, 1, -1, 1}]));
  EXPECT_EQ(7, (tensor[{2, 1, -1, 2}]));

  EXPECT_EQ(6, (tensor[{2, 1, 0, -1}]));
  EXPECT_EQ(5, (tensor[{2, 1, 0, 0}]));
  EXPECT_EQ(6, (tensor[{2, 1, 0, 1}]));
  EXPECT_EQ(5, (tensor[{2, 1, 0, 2}]));

  EXPECT_EQ(8, (tensor[{2, 1, 1, -1}]));
  EXPECT_EQ(7, (tensor[{2, 1, 1, 0}]));
  EXPECT_EQ(8, (tensor[{2, 1, 1, 1}]));
  EXPECT_EQ(7, (tensor[{2, 1, 1, 2}]));

  EXPECT_EQ(6, (tensor[{2, 1, 2, -1}]));
  EXPECT_EQ(5, (tensor[{2, 1, 2, 0}]));
  EXPECT_EQ(6, (tensor[{2, 1, 2, 1}]));
  EXPECT_EQ(5, (tensor[{2, 1, 2, 2}]));

  EXPECT_EQ(4, (tensor[{2, 2, -1, -1}]));
  EXPECT_EQ(3, (tensor[{2, 2, -1, 0}]));
  EXPECT_EQ(4, (tensor[{2, 2, -1, 1}]));
  EXPECT_EQ(3, (tensor[{2, 2, -1, 2}]));

  EXPECT_EQ(2, (tensor[{2, 2, 0, -1}]));
  EXPECT_EQ(1, (tensor[{2, 2, 0, 0}]));
  EXPECT_EQ(2, (tensor[{2, 2, 0, 1}]));
  EXPECT_EQ(1, (tensor[{2, 2, 0, 2}]));

  EXPECT_EQ(4, (tensor[{2, 2, 1, -1}]));
  EXPECT_EQ(3, (tensor[{2, 2, 1, 0}]));
  EXPECT_EQ(4, (tensor[{2, 2, 1, 1}]));
  EXPECT_EQ(3, (tensor[{2, 2, 1, 2}]));

  EXPECT_EQ(2, (tensor[{2, 2, 2, -1}]));
  EXPECT_EQ(1, (tensor[{2, 2, 2, 0}]));
  EXPECT_EQ(2, (tensor[{2, 2, 2, 1}]));
  EXPECT_EQ(1, (tensor[{2, 2, 2, 2}]));
}

}  // namespace sacred
