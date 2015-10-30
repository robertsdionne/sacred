package(
  default_visibility = ['//visibility:public'],
)

cc_library(
  name = 'checked_index',
  srcs = [
    'checked_index.hpp',
  ],
  deps = [
    ':identity_index',
    ':index_strategy',
    '//external:googlelog',
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'hashed_lookup',
  srcs = [
    'hashed_lookup.hpp',
  ],
  deps = [
    ':lookup_strategy',
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'identity_index',
  srcs = [
    'identity_index.hpp',
  ],
  deps = [
    ':index_strategy',
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'identity_lookup',
  srcs = [
    'identity_lookup.hpp',
  ],
  deps = [
    ':lookup_strategy',
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'index_strategy',
  srcs = [
    'index_strategy.hpp',
  ],
  deps = [
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'lookup_strategy',
  srcs = [
    'lookup_strategy.hpp',
  ],
  deps = [
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'masked_lookup',
  srcs = [
    'masked_lookup.hpp',
  ],
  deps = [
    ':identity_lookup',
    ':lookup_strategy',
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'mirrored_index',
  srcs = [
    'mirrored_index.hpp',
  ],
  deps = [
    ':index_strategy',
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'strides',
  srcs = [
    'strides.hpp',
  ],
  deps = [
    '//sacred:default_types',
  ],
)

cc_library(
  name = 'wrapped_index',
  srcs = [
    'wrapped_index.hpp',
  ],
  deps = [
    ':index_strategy',
    '//sacred:default_types',
  ],
)

cc_test(
  name = 'checked_index_test',
  srcs = [
    'checked_index_test.cpp',
  ],
  size = 'small',
  deps = [
    ':checked_index',
    '//third_party:gtest',
  ],
)

# cc_test(
#   name = 'hashed_lookup_test',
#   srcs = [
#     'hashed_lookup_test.cpp',
#   ],
#   size = 'small',
#   deps = [
#     ':hashed_lookup',
#     '//third_party:gtest',
#   ],
# )

cc_test(
  name = 'identity_index_test',
  srcs = [
    'identity_index_test.cpp',
  ],
  size = 'small',
  deps = [
    ':identity_index',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'identity_lookup_test',
  srcs =[
    'identity_lookup_test.cpp',
  ],
  size = 'small',
  deps = [
    ':identity_lookup',
    '//third_party:gtest',
  ],
)

# cc_test(
#   name = 'masked_lookup_test',
#   srcs = [
#     'masked_lookup_test.cpp',
#   ],
#   size = 'small',
#   deps = [
#     ':masked_lookup',
#     '//third_party:gtest',
#   ],
# )

cc_test(
  name = 'mirrored_index_test',
  srcs = [
    'mirrored_index_test.cpp',
  ],
  size = 'small',
  deps = [
    ':mirrored_index',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'strides_test',
  srcs = [
    'strides_test.cpp',
  ],
  size = 'small',
  deps = [
    ':strides',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'wrapped_index_test',
  srcs = [
    'wrapped_index_test.cpp',
  ],
  size = 'small',
  deps = [
    ':wrapped_index',
    '//third_party:gtest',
  ],
)
