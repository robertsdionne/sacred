#include <gtest/gtest.h>
#include <vector>

#include "strides.hpp"

namespace sacred { namespace indexing {

  using std::vector;

  TEST(Strides, CStyle) {
    EXPECT_EQ(vector<int>({30, 6, 1}), strides::CStyle<>({4, 5, 6}));
    EXPECT_EQ(vector<int>({32, 16, 8, 4, 2, 1}), strides::CStyle<>({2, 2, 2, 2, 2, 2}));
  }

}}  // namespaces
