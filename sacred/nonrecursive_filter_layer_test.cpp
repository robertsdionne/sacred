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

TEST(NonrecursiveFilterLayer, Backward) {
  auto output = Blob<float>({2, 2}, {
    348, 393,
    528, 573
  }, {
    1, 1,
    1, 1
  });
  auto bias = Blob<float>({2, 2});
  auto filter = Blob<float>({3, 3}, {
    9, 8, 7,
    6, 5, 4,
    3, 2, 1
  });
  auto layer = NonrecursiveFilterLayer<float>(bias, filter);
  auto input = Blob<float>({4, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  });
  layer.Backward(output, &input);
  EXPECT_EQ(Array<float>({2, 2}, {
    1, 1,
    1, 1
  }), bias.diff());
  EXPECT_EQ(Array<float>({3, 3}, {
    14, 18, 22,
    30, 34, 38,
    46, 50, 54
  }), filter.diff());
  EXPECT_EQ(Array<float>({4, 4}, {
    1, 3, 5, 3,
    5, 12, 16, 9,
    11, 24, 28, 15,
    7, 15, 17, 9
  }), input.diff());
}

TEST(NonrecursiveFilterLayer, Gradient) {
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
  auto target = Blob<float>({2, 2}, {
    349, 392,
    527, 574
  });
  auto layer = NonrecursiveFilterLayer<float>(bias, filter);

  constexpr auto kEpsilon = 1e-3f;

  for (auto i = 0; i < input.shape(0); ++i) {
    for (auto j = 0; j < input.shape(1); ++j) {
      auto output = Blob<float>({2, 2});
      input.diff() = Array<float>({4, 4});
      bias.diff() = Array<float>({2, 2});
      filter.diff() = Array<float>({3, 3});
      layer.Forward(input, &output);
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          output.diff({k, l}) = -delta;
        }
      }
      layer.Backward(output, &input);

      auto actual_partial_error_with_respect_to_input_ij = input.diff({i, j});
      auto original_input_ij = input.value({i, j});

      output = Blob<float>({2, 2});
      input.value({i, j}) = original_input_ij + kEpsilon;
      layer.Forward(input, &output);
      auto loss_1 = 0.0f;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          loss_1 += delta * delta / 2.0f;
        }
      }

      output = Blob<float>({2, 2});
      input.value({i, j}) = original_input_ij - kEpsilon;
      layer.Forward(input, &output);
      auto loss_0 = 0.0f;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          loss_0 += delta * delta / 2.0f;
        }
      }

      input.value({i, j}) = original_input_ij;

      auto expected_partial_error_with_respect_to_input_ij = (loss_1 - loss_0) / (2.0f * kEpsilon);
      EXPECT_NEAR(expected_partial_error_with_respect_to_input_ij, actual_partial_error_with_respect_to_input_ij, 1e-1);
    }
  }

  for (auto i = 0; i < filter.shape(0); ++i) {
    for (auto j = 0; j < filter.shape(1); ++j) {
      auto output = Blob<float>({2, 2});
      input.diff() = Array<float>({4, 4});
      bias.diff() = Array<float>({2, 2});
      filter.diff() = Array<float>({3, 3});
      layer.Forward(input, &output);
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          output.diff({k, l}) = -delta;
        }
      }
      layer.Backward(output, &input);

      auto actual_partial_error_with_respect_to_filter_ij = filter.diff({i, j});
      auto original_filter_ij = filter.value({i, j});

      output = Blob<float>({2, 2});
      filter.value({i, j}) = original_filter_ij + kEpsilon;
      layer.Forward(input, &output);
      auto loss_1 = 0.0f;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          loss_1 += delta * delta / 2.0f;
        }
      }

      output = Blob<float>({2, 2});
      filter.value({i, j}) = original_filter_ij - kEpsilon;
      layer.Forward(input, &output);
      auto loss_0 = 0.0f;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          loss_0 += delta * delta / 2.0f;
        }
      }

      filter.value({i, j}) = original_filter_ij;

      auto expected_partial_error_with_respect_to_filter_ij = (loss_1 - loss_0) / (2.0f * kEpsilon);
      EXPECT_NEAR(expected_partial_error_with_respect_to_filter_ij,
          actual_partial_error_with_respect_to_filter_ij, 1e-1);
    }
  }

  for (auto i = 0; i < bias.shape(0); ++i) {
    for (auto j = 0; j < bias.shape(1); ++j) {
      auto output = Blob<float>({2, 2});
      input.diff() = Array<float>({4, 4});
      bias.diff() = Array<float>({2, 2});
      filter.diff() = Array<float>({3, 3});
      layer.Forward(input, &output);
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          output.diff({k, l}) = -delta;
        }
      }
      layer.Backward(output, &input);

      auto actual_partial_error_with_respect_to_bias_ij = bias.diff({i, j});
      auto original_bias_ij = bias.value({i, j});

      output = Blob<float>({2, 2});
      bias.value({i, j}) = original_bias_ij + kEpsilon;
      layer.Forward(input, &output);
      auto loss_1 = 0.0f;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          loss_1 += delta * delta / 2.0f;
        }
      }

      output = Blob<float>({2, 2});
      bias.value({i, j}) = original_bias_ij - kEpsilon;
      layer.Forward(input, &output);
      auto loss_0 = 0.0f;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          loss_0 += delta * delta / 2.0f;
        }
      }

      bias.value({i, j}) = original_bias_ij;

      auto expected_partial_error_with_respect_to_bias_ij = (loss_1 - loss_0) / (2.0f * kEpsilon);
      EXPECT_NEAR(expected_partial_error_with_respect_to_bias_ij,
          actual_partial_error_with_respect_to_bias_ij, 1e-1);
    }
  }
}
