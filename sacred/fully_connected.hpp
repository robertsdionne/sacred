#ifndef SACRED_FULLY_CONNECTED_HPP_
#define SACRED_FULLY_CONNECTED_HPP_

#include <glog/logging.h>

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"
#include "testing.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class FullyConnected : public Operator<F> {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;

  FullyConnected(tensor_type &bias, tensor_type &weight) : bias_(bias), weight_(weight) {}

  virtual ~FullyConnected() = default;

  void operator ()(const tensors_type &in, const tensors_type &out) override {
    auto input = in.at(0), output = out.at(0);
    CHECK_LE(input->order(), 2);
    CHECK_GT(input->order(), 0);
    CHECK_LE(output->order(), 2);
    CHECK_GT(output->order(), 0);
    CHECK_EQ(input->order(), output->order());
    math_.GeneralMatrixMultiplication(*output, weight_, *input, F(0), F(1));
    math_.Add(*output, bias_, F(1), F(1));
  }

private:
  tensor_type &bias_, &weight_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_FULLY_CONNECTED_HPP_
