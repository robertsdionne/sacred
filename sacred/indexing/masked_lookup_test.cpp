#include <gtest/gtest.h>

#include "masked_lookup.hpp"

namespace sacred { namespace indexing {

TEST(MaskedLookup, Lookup) {
  auto masked_lookup = MaskedLookup<>();
  auto data = vector<float>({1, 2, 3, 4});

  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {-1, -1}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {-1, 0}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {-1, 1}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {-1, 2}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {0, -1}));
  EXPECT_EQ(1, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {0, 0}));
  EXPECT_EQ(2, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {0, 1}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {0, 2}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {1, -1}));
  EXPECT_EQ(3, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {1, 0}));
  EXPECT_EQ(4, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {1, 1}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {1, 2}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {2, -1}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {2, 0}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {2, 1}));
  EXPECT_EQ(0, masked_lookup.Lookup(data, data.size(), {2, 2}, {2, 1}, {2, 2}));
}

}}  // namespaces
