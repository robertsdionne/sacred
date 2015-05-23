#ifndef SACRED_MATH_HPP_
#define SACRED_MATH_HPP_

#include <iostream>

#include "array.hpp"
#include "checks.hpp"

namespace sacred {

  template <typename T>
  class Math {
  public:
    Math() = default;
    virtual ~Math() = default;

    void Convolve2(Array<T> &output, const Array<T> &filter, const Array<T> &input,
        const T output_coefficient, const T input_coefficient) {
      CHECK_STATE(input.shape(0) - filter.shape(0) + 1 == output.shape(0));
      CHECK_STATE(input.shape(1) - filter.shape(1) + 1 == output.shape(1));
      for (auto i = 0; i < output.shape(0); ++i) {
        for (auto j = 0; j < output.shape(1); ++j) {
          auto I = i + filter.shape(0) - 1;
          auto J = j + filter.shape(1) - 1;
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape(1); ++l) {
              current_output += input_coefficient * filter.at({k, l}) * input.at({I - k, J - l});
            }
          }
          output.at({i, j}) = current_output;
        }
      }
    }

    void GeneralMatrixMultiplication(Array<T> &output, const Array<T> &left, const Array<T> &right,
        const T output_coefficient, const T input_coefficient) {
      CHECK_STATE(left.shape(0) == output.shape(0));
      CHECK_STATE(right.shape(1) == output.shape(1));
      CHECK_STATE(left.shape(1) == right.shape(0));
      for (auto i = 0; i < output.shape(0); ++i) {
        for (auto j = 0; j < output.shape(1); ++j) {
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < right.shape(0); ++k) {
            current_output += input_coefficient * left.at({i, k}) * right.at({k, j});
          }
          output.at({i, j}) = current_output;
        }
      }
    }

    void RecurrentConvolve2(Array<T> &output, const Array<T> &filter,
        const T output_coefficient, const T input_coefficient) {
      for (auto j = 0; j < output.shape(1); ++j) {
        for (auto i = 0; i < output.shape(0); ++i) {
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape().at(1); ++l) {
              auto y = i + k - filter.shape(0) / 2;
              auto x = j + l - filter.shape(1);
              auto in = 0 <= y && y < output.shape(0) && 0 <= x;
              if (in) {
                current_output += input_coefficient * filter.at({k, l}) * output.at({y, x});
              }
            }
          }
          output.at({i, j}) = current_output;
        }
      }
    }
};

}  // namespace sacred

#endif  // SACRED_MATH_HPP_
