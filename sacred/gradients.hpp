#ifndef SACRED_GRADIENTS_HPP_
#define SACRED_GRADIENTS_HPP_

#include <functional>
#include <tuple>
#include <utility>

#include "default_types.hpp"
#include "dual.hpp"

namespace sacred {

using std::pair;
using std::tie;
using std::vector;

template <typename O, typename F = default_floating_point_type>
void TestGradients(
    const vector<pair<Tensor<Dual> *, Tensor<F> *>> &parameter_gradient_pairs,
    function<Tensor<Dual> *()> output_initializer,
    function<O *()> op_initializer,
    const Tensor<Dual> &input,
    const Tensor<Dual> &target) {

  for (auto pair : parameter_gradient_pairs) {
    Tensor<Dual> *parameter;
    Tensor<> *parameter_gradient;
    tie(parameter, parameter_gradient) = pair;

    for (auto i = 0; i < parameter->size(); ++i) {
      auto output = output_initializer();
      auto op = op_initializer();

      parameter->data(i) += 1_ɛ;

      (*op)(input, *output);

      auto loss = 0_ɛ;
      for (auto j = 0; j < target.size(); ++j) {
        auto delta = target.data(j) - output->data(j);
        loss += delta * delta / 2.0f;
      }

      parameter_gradient->data(i) = loss.dual;

      parameter->data(i).dual = 0;

      delete output;
      delete op;
    }
  }
}

}  // namespace sacred

#endif  // SACRED_GRADIENTS_HPP_
