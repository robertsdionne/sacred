#include <gtest/gtest.h>
#include <iostream>

#include "array.hpp"
#include "blob.hpp"
#include "dual.hpp"
#include "nonrecursive_filter_layer.hpp"

using namespace sacred;

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
    54, 50, 46, 
    38, 34, 30,
    22, 18, 14
  }), filter.diff());
  EXPECT_EQ(Array<float>({4, 4}, {
    1, 3, 5, 3,
    5, 12, 16, 9,
    11, 24, 28, 15,
    7, 15, 17, 9
  }), input.diff());
}

TEST(NonrecursiveFilterLayer, Gradient) {
  auto target = Blob<Dual>({2, 2}, {
    349, 392,
    527, 572
  });

  for (auto i = 0; i < 4; ++i) {
    for (auto j = 0; j < 4; ++j) {
      auto input = Blob<Dual>({4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
      });
      auto bias = Blob<Dual>({2, 2});
      auto filter = Blob<Dual>({3, 3}, {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
      });
      auto layer = NonrecursiveFilterLayer<Dual>(bias, filter);
      auto output = Blob<Dual>({2, 2});

      input.value({i, j}).dual = 1.0;
      layer.Forward(input, &output);

      auto loss = 0_ɛ;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          output.diff({k, l}) = -delta;
          loss += delta * delta / 2.0f;
        }
      }

      layer.Backward(output, &input);
      input.value({i, j}).dual = 0.0;

      auto actual_partial_error_with_respect_to_input_ij = input.diff({i, j}).real;
      auto expected_partial_error_with_respect_to_input_ij = loss.dual;

      EXPECT_NEAR(expected_partial_error_with_respect_to_input_ij, actual_partial_error_with_respect_to_input_ij, 1e-1);
    }
  }

  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) {
      auto input = Blob<Dual>({4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
      });
      auto bias = Blob<Dual>({2, 2});
      auto filter = Blob<Dual>({3, 3}, {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
      });
      auto layer = NonrecursiveFilterLayer<Dual>(bias, filter);
      auto output = Blob<Dual>({2, 2});

      filter.value({i, j}).dual = 1.0;
      layer.Forward(input, &output);

      std::cout << output.value() << std::endl;

      auto loss = 0_ɛ;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          output.diff({k, l}) = -delta;
          loss += delta * delta / 2.0f;
        }
      }

      layer.Backward(output, &input);
      filter.value({i, j}).dual = 0.0;

      auto actual_partial_error_with_respect_to_filter_ij = filter.diff({i, j}).real;
      auto expected_partial_error_with_respect_to_filter_ij = loss.dual;

      EXPECT_NEAR(expected_partial_error_with_respect_to_filter_ij,
          actual_partial_error_with_respect_to_filter_ij, 1e-1);
    }
  }

  for (auto i = 0; i < 2; ++i) {
    for (auto j = 0; j < 2; ++j) {
      auto input = Blob<Dual>({4, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
      });
      auto bias = Blob<Dual>({2, 2});
      auto filter = Blob<Dual>({3, 3}, {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
      });
      auto layer = NonrecursiveFilterLayer<Dual>(bias, filter);
      auto output = Blob<Dual>({2, 2});

      bias.value({i, j}).dual = 1.0;
      layer.Forward(input, &output);

      auto loss = 0_ɛ;
      for (auto k = 0; k < output.shape(0); ++k) {
        for (auto l = 0; l < output.shape(1); ++l) {
          auto delta = target.value({k, l}) - output.value({k, l});
          output.diff({k, l}) = -delta;
          loss += delta * delta / 2.0f;
        }
      }

      layer.Backward(output, &input);
      bias.value({i, j}).dual = 0.0;

      auto actual_partial_error_with_respect_to_bias_ij = bias.diff({i, j}).real;
      auto expected_partial_error_with_respect_to_bias_ij = loss.dual;

      EXPECT_NEAR(expected_partial_error_with_respect_to_bias_ij,
          actual_partial_error_with_respect_to_bias_ij, 1e-1);
    }
  }
}