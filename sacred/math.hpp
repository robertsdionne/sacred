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

    void Reconv(Array<T> &output, const Array<T> &filter) {
      for (auto i = 0; i < output.shape(0); ++i) {
        T current_output = output.at({i});
        auto I = i - 1;
        for (auto k = 0; k < filter.shape(0); ++k) {
          current_output += filter.at({k}) * output.at({I - k});
        }
        output.set({i}, current_output);
      }
    }

    void BackwardRecurrentConvolveFilter(Array<T> &filter_diff, const Array<T> &filter,
        const Array<T> &output_diff, const Array<T> &output) {
      auto scratch = Array<T>({filter.shape(0), output.shape(0), output.shape(1)});
      for (auto m = 0; m < filter.shape(0); ++m) {
        for (auto j = 0; j < scratch.shape(2); ++j) {
          for (auto i = 0; i < scratch.shape(1) + 1; ++i) {
            T current_output = T(0.0);
            current_output += output.at({i - filter.shape(0) / 2 + m, j - 1});
            for (auto k = 0; k < filter.shape(0); ++k) {
              for (auto l = 0; l < filter.shape(1); ++l) {
                current_output += filter.at({k, l}) * scratch.at({m, i - k + filter.shape(0) / 2, j - l - 1});
              }
            }
            scratch.add({m, i, j}, current_output);
          }
        }
      }
      for (auto j = 0; j < scratch.shape(2); ++j) {
        for (auto i = 0; i < scratch.shape(1); ++i) {
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape(1); ++l) {
              filter_diff.add({k, l}, scratch.at({filter.shape(0) - 1 - k, i, j}) * output_diff.at({i, j}));
            }
          }
        }
      }
      // std::cout << filter_diff << std::endl;
      // std::cout << filter << std::endl;
      // std::cout << output_diff << std::endl;
      // std::cout << output << std::endl;
      // std::cout << scratch << std::endl;
    }

    void BackwardReconv(Array<T> &filter_diff, const Array<T> &filter,
        const Array<T> &output_diff, const Array<T> &output) {
      auto scratch = Array<T>({output.shape(0)});
      for (auto i = 0; i < scratch.shape(0) - 1; ++i) {
        T current_output = T(0.0);
        auto I = i + 1;
        current_output += output.at({i});
        for (auto k = 0; k < filter.shape(0); ++k) {
          current_output += filter.at({k}) * scratch.at({i - k});
        }
        scratch.add({I}, current_output);
      }
      for (auto i = 0; i < scratch.shape(0) - 1; ++i) {
        auto I = i + 1;
        for (auto k = 0; k < filter.shape(0); ++k) {
          filter_diff.add({k}, scratch.at({I - k}) * output_diff.at({I}));
        }
      }
    }

    void Add(Array<T> &output, const Array<T> &input, const T output_coefficient, const T input_coefficient) const {
      CHECK_STATE(output.number_of_axes() == input.number_of_axes());
      for (auto i = 0; i < output.number_of_axes(); ++i) {
        CHECK_STATE(output.shape(i) == input.shape(i));
      }
      for (auto i = 0; i < output.count(); ++i) {
        output.data(i) = output_coefficient * output.data(i) + input_coefficient * input.data(i);
      }
    }

    void BroadcastAdd(Array<T> &output, const Array<T> &vector,
        const T output_coefficient, const T vector_coefficient) const {
      CHECK_STATE(output.shape(0) == vector.shape(0));
      for (auto i = 0; i < output.shape(0); ++i) {
        for (auto j = 0; j < output.shape(1); ++j) {
          output.set({i, j}, output_coefficient * output.at({i, j}) + vector_coefficient * vector.data(i));
        }
      }
    }

    void Sum(Array<T> &output, const Array<T> &input,
        const T output_coefficient, const T input_coefficient) {
      for (auto i = 0; i < input.shape(0); ++i) {
        T current_output = output_coefficient * output.data(i);
        for (auto j = 0; j < input.shape(1); ++j) {
          current_output += input_coefficient * input.at({i, j});
        }
        output.data(i) = current_output;
      }
    }

    void NarrowConvolve2(Array<T> &output, const Array<T> &filter, const Array<T> &input,
        const T output_coefficient, const T input_coefficient) const {
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
          output.set({i, j}, current_output);
        }
      }
    }

    void BackwardConvolve2(Array<T> &filter, const Array<T> &input, const Array<T> &output,
        const T filter_coefficient, const T output_coefficient) const {
      // CHECK_STATE(input.shape(0) + filter.shape(0) - 1 == output.shape(0));
      // CHECK_STATE(input.shape(1) + filter.shape(1) - 1 == output.shape(1));
      for (auto i = 0; i < filter.shape(0); ++i) {
        for (auto j = 0; j < filter.shape(1); ++j) {
          T current_filter = filter_coefficient * filter.at({i, j});
          for (auto k = 0; k < input.shape(0); ++k) {
            for (auto l = 0; l < input.shape(1); ++l) {
              auto I = i - input.shape(0) / 2;
              auto J = j - 1;
              current_filter += output_coefficient * input.at({k, l}) * output.at({I + k, J + l});
            }
          }
          filter.set({i, j}, current_filter);
        }
      }
    }

    void BackwardNarrowConvolve2(Array<T> &filter, const Array<T> &input, const Array<T> &output,
        const T filter_coefficient, const T output_coefficient) const {
      CHECK_STATE(input.shape(0) - filter.shape(0) + 1 == output.shape(0));
      CHECK_STATE(input.shape(1) - filter.shape(1) + 1 == output.shape(1));
      for (auto i = 0; i < filter.shape(0); ++i) {
        for (auto j = 0; j < filter.shape(1); ++j) {
          T current_filter = filter_coefficient * filter.at({i, j});
          for (auto k = 0; k < input.shape(0); ++k) {
            for (auto l = 0; l < input.shape(1); ++l) {
              auto I = i - input.shape(0) / 2;
              auto J = j - input.shape(1) / 2;
              current_filter += output_coefficient * input.at({k, l}) * output.at({I + k, J + l});
            }
          }
          filter.set({i, j}, current_filter);
        }
      }
    }

    void WideConvolve2(Array<T> &output, const Array<T> &filter, const Array<T> &input,
        const T output_coefficient, const T input_coefficient) const {
      CHECK_STATE(input.shape(0) + filter.shape(0) - 1 == output.shape(0));
      CHECK_STATE(input.shape(1) + filter.shape(1) - 1 == output.shape(1));
      for (auto i = 0; i < output.shape(0); ++i) {
        for (auto j = 0; j < output.shape(1); ++j) {
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape(1); ++l) {
              current_output += input_coefficient * filter.at({k, l}) * input.at({i - k, j - l});
            }
          }
          output.set({i, j}, current_output);
        }
      }
    }

    void BackwardWideConvolve2(Array<T> &output, const Array<T> &filter, const Array<T> &input,
        const T output_coefficient, const T input_coefficient) const {
      CHECK_STATE(input.shape(0) + filter.shape(0) - 1 == output.shape(0));
      CHECK_STATE(input.shape(1) + filter.shape(1) - 1 == output.shape(1));
      for (auto i = 0; i < output.shape(0); ++i) {
        for (auto j = 0; j < output.shape(1); ++j) {
          auto I = i - filter.shape(0) + 1;
          auto J = j - filter.shape(1) + 1;
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape(1); ++l) {
              current_output += input_coefficient * filter.at({k, l}) * input.at({I + k, J + l});
            }
          }
          output.set({i, j}, current_output);
        }
      }
    }

    void RecurrentConvolve2(Array<T> &output, const Array<T> &filter,
        const T output_coefficient, const T input_coefficient) const {
      for (auto j = 0; j < output.shape(1); ++j) {
        for (auto i = 0; i < output.shape(0); ++i) {
          auto I = i + filter.shape(0) / 2;
          auto J = j - 1;
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape(1); ++l) {
              current_output += input_coefficient * filter.at({k, l}) * output.at({I - k, J - l});
            }
          }
          output.set({i, j}, current_output);
        }
      }
    }

    void BackwardRecurrentConvolve2(Array<T> &output, const Array<T> &filter,
        const T output_coefficient, const T input_coefficient) const {
      for (auto j = output.shape(1) - 1; j >= 0; --j) {
        for (auto i = 0; i < output.shape(0); ++i) {
          auto I = i - filter.shape(0) / 2;
          auto J = j + 1;
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < filter.shape(0); ++k) {
            for (auto l = 0; l < filter.shape(1); ++l) {
              current_output += input_coefficient * filter.at({k, l}) * output.at({I + k, J + l});
            }
          }
          output.set({i, j}, current_output);
        }
      }
    }

    void GeneralMatrixMultiplication(Array<T> &output, const Array<T> &left, const Array<T> &right,
        const T output_coefficient, const T input_coefficient) const {
      CHECK_STATE(left.shape(0) == output.shape(0));
      CHECK_STATE(right.shape(1) == output.shape(1));
      CHECK_STATE(left.shape(1) == right.shape(0));
      for (auto i = 0; i < output.shape(0); ++i) {
        for (auto j = 0; j < output.shape(1); ++j) {
          T current_output = output_coefficient * output.at({i, j});
          for (auto k = 0; k < right.shape(0); ++k) {
            current_output += input_coefficient * left.at({i, k}) * right.at({k, j});
          }
          output.set({i, j}, current_output);
        }
      }
    }
  };

}  // namespace sacred

#endif  // SACRED_MATH_HPP_
