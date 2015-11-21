#ifndef SACRED_MATH_HPP_
#define SACRED_MATH_HPP_

#include <glog/logging.h>
#include <iostream>

#include "default_types.hpp"
#include "indexing/identity_index.hpp"
#include "indexing/masked_lookup.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class Math {
public:
  using tensor_type = Tensor<F>;

  Math() = default;

  virtual ~Math() = default;

  void Add(tensor_type &output, const tensor_type &input, const F output_coefficient, const F input_coefficient) const {
    CHECK(output.number_of_axes() == input.number_of_axes());
    for (auto i = 0; i < output.number_of_axes(); ++i) {
      CHECK(output.shape().at(i) == input.shape().at(i));
    }
    for (auto i = 0; i < output.size(); ++i) {
      output.data(i) = output_coefficient * output.data(i) + input_coefficient * input.data(i);
    }
  }

  void BroadcastAdd(tensor_type &output, const tensor_type &vector,
      const F output_coefficient, const F vector_coefficient) const {
    CHECK(output.shape().at(0) == vector.shape().at(0));
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

  void GeneralMatrixMultiplication(tensor_type &output, const tensor_type &left, const tensor_type &right,
      const F output_coefficient, const F input_coefficient) const {
    CHECK(left.shape().at(0) == output.shape().at(0));
    CHECK(right.shape().at(1) == output.shape().at(1));
    CHECK(left.shape().at(1) == right.shape().at(0));
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

  void GeneralMatrixMultiplicationTransposeNormal(tensor_type &output,
      const tensor_type &left, const tensor_type &right, const F output_coefficient, const F input_coefficient) const {
    CHECK(true);
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < right.shape().at(0); ++k) {
          current_output += input_coefficient * left.at({k, i}) * right.at({k, j});
        }
        output.set({i, j}, current_output);
      }
    }
  }

  void GeneralMatrixMultiplicationNormalTranspose(tensor_type &output,
      const tensor_type &left, const tensor_type &right, const F output_coefficient, const F input_coefficient) const {
    CHECK(true);
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < right.shape().at(1); ++k) {
          current_output += input_coefficient * left.at({i, k}) * right.at({j, k});
        }
        output.set({i, j}, current_output);
      }
    }
  }

  void GeneralMatrixMultiplicationTransposeTranspose(tensor_type &output,
      const tensor_type &left, const tensor_type &right, const F output_coefficient, const F input_coefficient) const {
    CHECK(true);
    for (auto i = 0; i < output.shape().at(0); ++i) {
      for (auto j = 0; j < output.shape().at(1); ++j) {
        F current_output = output_coefficient * output.at({i, j});
        for (auto k = 0; k < right.shape().at(1); ++k) {
          current_output += input_coefficient * left.at({k, i}) * right.at({j, k});
        }
        output.set({i, j}, current_output);
      }
    }
  }
};

}  // namespace sacred

#endif  // SACRED_MATH_HPP_
