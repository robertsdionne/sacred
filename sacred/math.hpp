#ifndef SACRED_MATH_HPP_
#define SACRED_MATH_HPP_

#include <iostream>

#include "default_types.hpp"
#include "checks.hpp"
#include "identity_index.hpp"
#include "masked_lookup.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class Math {
public:
  using tensor_type = Tensor<F>;

  Math() = default;

  virtual ~Math() = default;

  void Reconv(tensor_type &output, const tensor_type &filter) {
    for (auto i = 0; i < output.shape().at(0); ++i) {
      F current_output = output.at({i});
      auto I = i - 1;
      for (auto k = 0; k < filter.shape().at(0); ++k) {
        current_output += filter.at({k}) * output.at({I - k});
      }
      output.set({i}, current_output);
    }
  }

  void BackwardRecurrentConvolveFilter(tensor_type &filter_diff, const tensor_type &filter,
      const tensor_type &output_diff, const tensor_type &output) {
    auto scratch = tensor_type({filter.shape().at(0), output.shape().at(0), output.shape().at(1)});
    for (auto m = 0; m < filter.shape().at(0); ++m) {
      for (auto j = 0; j < scratch.shape().at(2); ++j) {
        for (auto i = 0; i < scratch.shape().at(1) + 1; ++i) {
          F current_output = F(0);
          current_output += output.at({i - filter.shape().at(0) / 2 + m, j - 1});
          for (auto k = 0; k < filter.shape().at(0); ++k) {
            for (auto l = 0; l < filter.shape().at(1); ++l) {
              current_output += filter.at({k, l}) * scratch.at({m, i - k + filter.shape().at(0) / 2, j - l - 1});
            }
          }
          scratch.add({m, i, j}, current_output);
        }
      }
    }
    for (auto j = 0; j < scratch.shape().at(2); ++j) {
      for (auto i = 0; i < scratch.shape().at(1); ++i) {
        for (auto k = 0; k < filter.shape().at(0); ++k) {
          for (auto l = 0; l < filter.shape().at(1); ++l) {
            filter_diff.add({k, l}, scratch.at({filter.shape().at(0) - 1 - k, i, j - l}) * output_diff.at({i, j}));
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

  void BackwardReconv(tensor_type &filter_diff, const tensor_type &filter,
      const tensor_type &output_diff, const tensor_type &output) {
    auto scratch = tensor_type({output.shape().at(0)});
    for (auto i = 0; i < scratch.shape().at(0) - 1; ++i) {
      F current_output = F(0);
      auto I = i + 1;
      current_output += output.at({i});
      for (auto k = 0; k < filter.shape().at(0); ++k) {
        current_output += filter.at({k}) * scratch.at({i - k});
      }
      scratch.add({I}, current_output);
    }
    for (auto i = 0; i < scratch.shape().at(0) - 1; ++i) {
      auto I = i + 1;
      for (auto k = 0; k < filter.shape().at(0); ++k) {
        filter_diff.add({k}, scratch.at({I - k}) * output_diff.at({I}));
      }
    }
  }

  void Add(tensor_type &output, const tensor_type &input, const F output_coefficient, const F input_coefficient) const {
    CHECK_STATE(output.number_of_axes() == input.number_of_axes());
    for (auto i = 0; i < output.number_of_axes(); ++i) {
      CHECK_STATE(output.shape().at(i) == input.shape().at(i));
    }
    for (auto i = 0; i < output.count(); ++i) {
      output.data(i) = output_coefficient * output.data(i) + input_coefficient * input.data(i);
    }
  }

  void BroadcastAdd(tensor_type &output, const tensor_type &vector,
      const F output_coefficient, const F vector_coefficient) const {
    CHECK_STATE(output.shape().at(0) == vector.shape().at(0));
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        output.set({i, j}, output_coefficient * output.at({i, j}) + vector_coefficient * vector.data(i));
      }
    }
  }

  void Sum(tensor_type &output, const tensor_type &input,
      const F output_coefficient, const F input_coefficient) {
    for (auto i = 0; i < input.shape().at(0); ++i) {
      F current_output = output_coefficient * output.data(i);
      for (auto j = 0; j < input.shape().at(1); ++j) {
        current_output += input_coefficient * input.at({i, j});
      }
      output.data(i) = current_output;
    }
  }

  void NarrowConvolve2(tensor_type &output, const tensor_type &filter, const tensor_type &input,
      const F output_coefficient, const F input_coefficient) const {
    CHECK_STATE(input.shape().at(0) - filter.shape().at(0) + 1 == output.shape().at(0));
    CHECK_STATE(input.shape().at(1) - filter.shape().at(1) + 1 == output.shape().at(1));
    for (auto i = 0; i < output.shape().at(0); ++i) {
      std::cout << ".";
      std::cout.flush();
      for (auto j = 0; j < output.shape().at(1); ++j) {
        auto I = i + filter.shape().at(0) - 1;
        auto J = j + filter.shape().at(1) - 1;
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < filter.shape().at(0); ++k) {
          for (auto l = 0; l < filter.shape().at(1); ++l) {
            current_output += input_coefficient * filter.at({k, l}) * input.at({I - k, J - l});
          }
        }
        output.set({i, j}, current_output);
      }
    }
    std::cout << std::endl;
  }

  void BackwardConvolve2(tensor_type &filter, const tensor_type &input, const tensor_type &output,
      const F filter_coefficient, const F output_coefficient) const {
    // CHECK_STATE(input.shape().at(0) + filter.shape().at(0) - 1 == output.shape().at(0));
    // CHECK_STATE(input.shape().at(1) + filter.shape().at(1) - 1 == output.shape().at(1));
    for (auto i = 0; i < filter.shape().at(0); ++i) {
      for (auto j = 0; j < filter.shape().at(1); ++j) {
        F current_filter = filter_coefficient * filter.at({i, j});
        for (auto k = 0; k < input.shape().at(0); ++k) {
          for (auto l = 0; l < input.shape().at(1); ++l) {
            auto I = i - input.shape().at(0) / 2;
            auto J = j - 1;
            current_filter += output_coefficient * input.at({k, l}) * output.at({I + k, J + l});
          }
        }
        filter.set({i, j}, current_filter);
      }
    }
  }

  void BackwardNarrowConvolve2(tensor_type &filter, const tensor_type &input, const tensor_type &output,
      const F filter_coefficient, const F output_coefficient) const {
    CHECK_STATE(input.shape().at(0) - filter.shape().at(0) + 1 == output.shape().at(0));
    CHECK_STATE(input.shape().at(1) - filter.shape().at(1) + 1 == output.shape().at(1));
    for (auto i = 0; i < filter.shape().at(0); ++i) {
      for (auto j = 0; j < filter.shape().at(1); ++j) {
        F current_filter = filter_coefficient * filter.at({i, j});
        for (auto k = 0; k < input.shape().at(0); ++k) {
          for (auto l = 0; l < input.shape().at(1); ++l) {
            auto I = i - input.shape().at(0) / 2;
            auto J = j - input.shape().at(1) / 2;
            current_filter += output_coefficient * input.at({k, l}) * output.at({I + k, J + l});
          }
        }
        filter.set({i, j}, current_filter);
      }
    }
  }

  void WideConvolve2(tensor_type &output, const tensor_type &filter, const tensor_type &input,
      const F output_coefficient, const F input_coefficient) const {
    CHECK_STATE(input.shape().at(0) + filter.shape().at(0) - 1 == output.shape().at(0));
    CHECK_STATE(input.shape().at(1) + filter.shape().at(1) - 1 == output.shape().at(1));
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < filter.shape().at(0); ++k) {
          for (auto l = 0; l < filter.shape().at(1); ++l) {
            current_output += input_coefficient * filter.at({k, l}) * input.template at<IdentityIndex<int>, MaskedLookup<F>>({i - k, j - l});
          }
        }
        output.set({i, j}, current_output);
      }
    }
  }

  void BackwardWideConvolve2(tensor_type &output, const tensor_type &filter, const tensor_type &input,
      const F output_coefficient, const F input_coefficient) const {
    CHECK_STATE(input.shape().at(0) + filter.shape().at(0) - 1 == output.shape().at(0));
    CHECK_STATE(input.shape().at(1) + filter.shape().at(1) - 1 == output.shape().at(1));
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        auto I = i - filter.shape().at(0) + 1;
        auto J = j - filter.shape().at(1) + 1;
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < filter.shape().at(0); ++k) {
          for (auto l = 0; l < filter.shape().at(1); ++l) {
            current_output += input_coefficient * filter.at({k, l}) * input.template at<IdentityIndex<int>, MaskedLookup<F>>({I + k, J + l});
          }
        }
        output.set({i, j}, current_output);
      }
    }
  }

  void RecurrentConvolve2(tensor_type &output, const tensor_type &filter,
      const F output_coefficient, const F input_coefficient) const {
    for (auto j = 0; j < output.shape().at(1); ++j) {
      std::cout << ".";
      std::cout.flush();
      for (auto i = 0; i < output.shape().at(0); ++i) {
        auto I = i + filter.shape().at(0) / 2;
        auto J = j - 1;
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < filter.shape().at(0); ++k) {
          for (auto l = 0; l < filter.shape().at(1); ++l) {
            current_output += input_coefficient * filter.at({k, l}) * output.template at<IdentityIndex<int>, MaskedLookup<F>>({I - k, J - l});
          }
        }
        output.set({i, j}, current_output);
      }
    }
    std::cout << std::endl;
  }

  void BackwardRecurrentConvolve2(tensor_type &output, const tensor_type &filter,
      const F output_coefficient, const F input_coefficient) const {
    for (auto j = output.shape().at(1) - 1; j >= 0; --j) {
      for (auto i = 0; i < output.shape().at(0); ++i) {
        auto I = i - filter.shape().at(0) / 2;
        auto J = j + 1;
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < filter.shape().at(0); ++k) {
          for (auto l = 0; l < filter.shape().at(1); ++l) {
            current_output += input_coefficient * filter.at({k, l}) * output.template at<IdentityIndex<int>, MaskedLookup<F>>({I + k, J + l});
          }
        }
        output.set({i, j}, current_output);
      }
    }
  }

  void GeneralMatrixMultiplication(tensor_type &output, const tensor_type &left, const tensor_type &right,
      const F output_coefficient, const F input_coefficient) const {
    CHECK_STATE(left.shape().at(0) == output.shape().at(0));
    CHECK_STATE(right.shape().at(1) == output.shape().at(1));
    CHECK_STATE(left.shape().at(1) == right.shape().at(0));
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < right.shape().at(0); ++k) {
          current_output += input_coefficient * left.at({i, k}) * right.at({k, j});
        }
        output.set({i, j}, current_output);
      }
    }
  }
};

}  // namespace sacred

#endif  // SACRED_MATH_HPP_
