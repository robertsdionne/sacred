#ifndef SACRED_MATH_HPP_
#define SACRED_MATH_HPP_

#include <iostream>

namespace sacred {

  template <typename T>
  class Math {
  public:
    Math() = default;
    virtual ~Math() = default;

    inline T &Access(T *input,
        const int input_height, const int input_width,
        const int row, const int column) {
      return input[row * input_width + column];
    }

    inline const T &Access(const T *input,
        const int input_height, const int input_width,
        const int row, const int column) {
      return input[row * input_width + column];
    }

    void Convolve2(T *output, const T *filter, const T *input,
        const T output_coefficient, const T input_coefficient,
        const int filter_height, const int filter_width,
        const int input_height, const int input_width) {
      const auto output_height = input_height - filter_height + 1;
      const auto output_width = input_width - filter_width + 1;
      for (auto i = 0; i < output_height; ++i) {
        for (auto j = 0; j < output_width; ++j) {
          T current_output = output_coefficient * Access(output, output_height, output_width, i, j);
          for (auto k = 0; k < filter_height; ++k) {
            for (auto l = 0; l < filter_width; ++l) {
              current_output += input_coefficient
                  * Access(filter, filter_height, filter_width, k, l)
                  * Access(input, input_height, input_width, i + k, j + l);
            }
          }
          Access(output, output_height, output_width, i, j) = current_output;
        }
      }
    }

    void GeneralMatrixMultiplication(T *output, const T *left, const T *right,
        const T output_coefficient, const T input_coefficient,
        const int output_height, const int input_height, const int right_height) {
      for (auto i = 0; i < output_height; ++i) {
        for (auto j = 0; j < input_height; ++j) {
          T current_output = output_coefficient * Access(output, output_height, input_height, i, j);
          for (auto k = 0; k < right_height; ++k) {
            current_output += input_coefficient * Access(left, output_height, right_height, i, k)
                * Access(right, right_height, input_height, k, j);
          }
          Access(output, output_height, input_height, i, j) = current_output;
        }
      }
    }

    void Print(const T *input, const int input_height, const int input_width) {
      for (auto i = 0; i < input_height; ++i) {
        for (auto j = 0; j < input_width; ++j) {
          std::cout << Access(input, input_height, input_width, i, j);
          if (j < input_width - 1) {
            std::cout << u8" ";
          }
        }
        std::cout << std::endl;
      }
    }

    void RecurrentConvolve2(T *output, const T *filter,
        const T output_coefficient, const T input_coefficient,
        const int output_height, const int output_width,
        const int filter_height, const int filter_width) {
      for (auto j = 0; j < output_width; ++j) {
        for (auto i = 0; i < output_height; ++i) {
          T current_output = output_coefficient * Access(output, output_height, output_width, i, j);
          for (auto k = 0; k < filter_height; ++k) {
            for (auto l = 0; l < filter_width; ++l) {
              auto y = i + k - filter_height / 2;
              auto x = j + l - filter_width;
              auto in = 0 <= y && y < output_height && 0 <= x;
              if (in) {
                current_output += input_coefficient
                    * Access(filter, filter_height, filter_width, k, l)
                    * Access(output, output_height, output_width, y, x);
              }
            }
          }
          Access(output, output_height, output_width, i, j) = current_output;
        }
      }
    }
};

}  // namespace sacred

#endif  // SACRED_MATH_HPP_
