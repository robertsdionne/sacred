#include <gtest/gtest.h>

#include "mirrored_index.hpp"

namespace sacred { namespace indexing {

TEST(MirroredIndex, Transform) {
  auto mirrored_index = MirroredIndex<>();

  EXPECT_EQ(2, mirrored_index.Transform({3}, {1}, {-6}).at(0));
  EXPECT_EQ(1, mirrored_index.Transform({3}, {1}, {-5}).at(0));
  EXPECT_EQ(0, mirrored_index.Transform({3}, {1}, {-4}).at(0));
  EXPECT_EQ(1, mirrored_index.Transform({3}, {1}, {-3}).at(0));
  EXPECT_EQ(2, mirrored_index.Transform({3}, {1}, {-2}).at(0));
  EXPECT_EQ(1, mirrored_index.Transform({3}, {1}, {-1}).at(0));
  EXPECT_EQ(0, mirrored_index.Transform({3}, {1}, {0}).at(0));
  EXPECT_EQ(1, mirrored_index.Transform({3}, {1}, {1}).at(0));
  EXPECT_EQ(2, mirrored_index.Transform({3}, {1}, {2}).at(0));
  EXPECT_EQ(1, mirrored_index.Transform({3}, {1}, {3}).at(0));
  EXPECT_EQ(0, mirrored_index.Transform({3}, {1}, {4}).at(0));
  EXPECT_EQ(1, mirrored_index.Transform({3}, {1}, {5}).at(0));
  EXPECT_EQ(2, mirrored_index.Transform({3}, {1}, {6}).at(0));
}

}}  // namespaces
