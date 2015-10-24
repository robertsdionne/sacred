#include <gtest/gtest.h>

#include "tensor_train.hpp"

namespace sacred {

TEST(TensorTrain, Initialize) {
  auto tensor_train = TensorTrain<>();
}

}  // namespace sacred
