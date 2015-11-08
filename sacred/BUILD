package(
  default_visibility = ['//visibility:public'],
)

cc_library(
  name = 'default_types',
  srcs = [
    'default_types.hpp',
  ],
)

cc_library(
  name = 'dual',
  srcs = [
    'dual.hpp',
  ],
)

cc_library(
  name = 'fully_connected',
  srcs = [
    'fully_connected.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
    ':testing',
    '//external:googlelog',
  ],
)

cc_library(
  name = 'fully_connected_gradient',
  srcs = [
    'fully_connected_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
  ],
)

cc_test(
  name = 'fully_connected_gradient_test',
  srcs = [
    'fully_connected_gradient_test.cpp',
  ],
  size = 'small',
  deps = [
    ':fully_connected_gradient',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'fully_connected_test',
  srcs = [
    'fully_connected_test.cpp',
  ],
  size = 'small',
  deps = [
    ':dual',
    ':fully_connected',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'functional',
  srcs = [
    'functional.hpp',
  ],
)

# cc_library(
#   name = 'lookup_table',
#   srcs = [
#     'lookup_table.hpp'
#   ],
#   deps = [
#     ':blob',
#   ],
# )

cc_library(
  name = 'math',
  srcs = [
    'math.hpp',
  ],
  deps = [
    ':default_types',
    ':tensor',
    '//external:googlelog',
    '//sacred/indexing:identity_index',
    '//sacred/indexing:masked_lookup',
  ],
)

# cc_library(
#   name = 'memory_mapped_lookup_table',
#   srcs = [
#     'memory_mapped_lookup_table.hpp',
#   ],
#   deps = [
#     ':blob',
#     ':lookup_table',
#   ],
# )

cc_library(
  name = 'nonrecursive_filter',
  srcs = [
    'nonrecursive_filter.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
    ':tensor',
  ],
)

cc_library(
  name = 'nonrecursive_filter_gradient',
  srcs = [
    'nonrecursive_filter_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
    ':tensor',
  ],
)

cc_library(
  name = 'operator',
  srcs = [
    'operator.hpp',
  ],
  deps = [
    ':default_types',
    ':tensor',
  ],
)

cc_library(
  name = 'rectifier',
  srcs = [
    'rectifier.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
    '//external:googlelog',
  ],
)

cc_test(
  name = 'rectifier_test',
  srcs = [
    'rectifier_test.cpp',
  ],
  size = 'small',
  deps = [
    ':rectifier',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'recursive_filter',
  srcs = [
    'recursive_filter.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
    ':tensor',
  ],
)

cc_library(
  name = 'recursive_filter_gradient',
  srcs = [
    'recursive_filter_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
    ':tensor',
  ],
)

cc_library(
  name = 'tensor',
  srcs = [
    'tensor.hpp',
  ],
  deps = [
    ':default_types',
    ':functional',
    ':tensor_interface',
    ':testing',
    '//external:googlelog',
    '//sacred/indexing:checked_index',
    '//sacred/indexing:identity_lookup',
    '//sacred/indexing:index_strategy',
    '//sacred/indexing:lookup_strategy',
    '//sacred/indexing:strides',
    '//sacred/indexing:wrapped_index',
  ],
)

cc_library(
  name = 'tensor_interface',
  srcs = [
    'tensor_interface.hpp',
  ],
  deps = [
    ':default_types',
    ':functional',
    ':testing',
  ],
)

cc_library(
  name = 'tensor_train',
  srcs = [
    'tensor_train.hpp',
  ],
  deps = [
    ':default_types',
    ':functional',
    ':tensor_interface',
    ':tensor',
    '//sacred/indexing:checked_index',
    '//sacred/indexing:identity_lookup',
    '//sacred/indexing:index_strategy',
    '//sacred/indexing:lookup_strategy',
    '//sacred/indexing:wrapped_index',
  ],
)

cc_library(
  name = 'testing',
  srcs = [
    'testing.hpp',
  ],
)

cc_test(
  name = 'math_test',
  srcs = [
    'math_test.cpp',
  ],
  size = 'small',
  deps = [
    ':dual',
    ':math',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'nonrecursive_filter_gradient_test',
  srcs = [
    'nonrecursive_filter_gradient_test.cpp',
  ],
  size = 'small',
  deps = [
    ':dual',
    ':nonrecursive_filter_gradient',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'nonrecursive_filter_test',
  srcs = [
    'nonrecursive_filter_test.cpp',
  ],
  size = 'small',
  deps = [
    ':nonrecursive_filter',
    ':tensor',
    '//third_party:gtest',
  ],
)

# cc_test(
#   name = 'reconv_test',
#   srcs = [
#     'reconv_test.cpp',
#   ],
#   size = 'small',
#   deps = [
#     ':array',
#     ':dual',
#     ':math',
#     '//third_party:gtest',
#   ],
# )

# cc_test(
#   name = 'recursive_filter_gradient_test',
#   srcs = [
#     'recursive_filter_gradient_test.cpp',
#   ],
#   size = 'small',
#   deps = [
#     ':dual',
#     ':recursive_filter_gradient',
#     ':tensor',
#     '//third_party:gtest',
#   ],
# )

cc_test(
  name = 'recursive_filter_test',
  srcs = [
    'recursive_filter_test.cpp',
  ],
  size = 'small',
  deps = [
    ':recursive_filter',
    ':tensor',
    '//third_party:gtest',
  ],
)

# cc_test(
#   name = 'stacked_test',
#   srcs = [
#     'stacked_test.cpp',
#   ],
#   size = 'small',
#   deps = [
#     ':array',
#     ':blob',
#     ':dual',
#     ':nonrecursive_filter',
#     ':nonrecursive_filter_gradient',
#     ':recursive_filter_layer',
#     '//third_party:gtest',
#   ],
# )

cc_test(
  name = 'tensor_test',
  srcs = [
    'tensor_test.cpp',
  ],
  size = 'small',
  deps = [
    ':tensor',
    '//sacred/indexing:mirrored_index',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'tensor_train_test',
  srcs = [
    'tensor_train_test.cpp',
  ],
  size = 'small',
  deps = [
    ':tensor_train',
    '//third_party:gtest',
  ],
)

# cc_binary(
#   name = 'sacred',
#   srcs = [
#     'sacred.cpp',
#   ],
#   deps = [
#     '//external:googleflags',
#   ],
# )
