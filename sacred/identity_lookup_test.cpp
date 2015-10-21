#include <gtest/gtest.h>

#include "identity_lookup.hpp"

namespace sacred {

TEST(IdentityLookup, Lookup) {
  auto identity_lookup = IdentityLookup<>();
  auto data = vector<float>({0, 1, 2});

  EXPECT_EQ(0, identity_lookup.Lookup(data, data.size(), {3}, {1}, {0}));
  EXPECT_EQ(1, identity_lookup.Lookup(data, data.size(), {3}, {1}, {1}));
  EXPECT_EQ(2, identity_lookup.Lookup(data, data.size(), {3}, {1}, {2}));
}

}  // namespace sacred
