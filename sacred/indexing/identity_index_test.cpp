#include <gtest/gtest.h>

#include "identity_index.hpp"

namespace sacred { namespace indexing {

TEST(IdentityIndex, Transform) {
  auto identity_index = IdentityIndex<>();

  EXPECT_EQ(-1, identity_index.Transform({3}, {1}, {-1}).at(0));
  EXPECT_EQ(0, identity_index.Transform({3}, {1}, {0}).at(0));
  EXPECT_EQ(1, identity_index.Transform({3}, {1}, {1}).at(0));
  EXPECT_EQ(2, identity_index.Transform({3}, {1}, {2}).at(0));
  EXPECT_EQ(3, identity_index.Transform({3}, {1}, {3}).at(0));
}

}}  // namespaces
