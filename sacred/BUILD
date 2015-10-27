cc_library(
  name = 'array',
  srcs = [
    'array.hpp',
  ],
  deps = [
    ':checks',
    ':clipped_index_strategy',
    ':dual',
    ':hashed_index_strategy',
    ':index_strategy',
    ':tiled_index_strategy',
  ],
)

cc_library(
  name = 'blob',
  srcs = [
    'blob.hpp',
  ],
  deps = [
    ':array',
    ':clipped_index_strategy',
    ':index_strategy',
  ],
)

cc_library(
  name = 'checked_index',
  srcs = [
    'checked_index.hpp',
  ],
  deps = [
    ':checks',
    ':default_types',
    ':identity_index',
    ':index_strategy',
  ],
)

cc_library(
  name = 'checks',
  srcs = [
    'checks.hpp',
  ],
)

cc_library(
  name = 'clipped_index_strategy',
  srcs = [
    'clipped_index_strategy.hpp',
  ],
  deps = [
    ':checks',
    ':index_strategy',
  ],
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
  name = 'functional',
  srcs = [
    'functional.hpp',
  ],
)

cc_library(
  name = 'hashed_index_strategy',
  srcs = [
    'hashed_index_strategy.hpp',
  ],
  deps = [
    ':checks',
    ':index_strategy',
  ],
)

cc_library(
  name = 'hashed_lookup',
  srcs = [
    'hashed_lookup.hpp',
  ],
  deps = [
    ':default_types',
    ':lookup_strategy',
  ],
)

cc_library(
  name = 'identity_index',
  srcs = [
    'identity_index.hpp',
  ],
  deps = [
    ':default_types',
    ':index_strategy',
  ],
)

cc_library(
  name = 'identity_lookup',
  srcs = [
    'identity_lookup.hpp',
  ],
  deps = [
    ':default_types',
    ':lookup_strategy',
  ],
)

cc_library(
  name = 'index_strategy',
  srcs = [
    'index_strategy.hpp',
  ],
  deps = [
    ':default_types',
  ],
)

cc_library(
  name = 'lookup_strategy',
  srcs = [
    'lookup_strategy.hpp',
  ],
  deps = [
    ':default_types',
  ],
)

cc_library(
  name = 'lookup_table',
  srcs = [
    'lookup_table.hpp'
  ],
  deps = [
    ':blob',
  ],
)

cc_library(
  name = 'masked_lookup',
  srcs = [
    'masked_lookup.hpp',
  ],
  deps = [
    ':default_types',
    ':identity_lookup',
    ':lookup_strategy',
  ],
)

cc_library(
  name = 'math',
  srcs = [
    'math.hpp',
  ],
  deps = [
    ':checks',
    ':default_types',
    ':identity_index',
    ':masked_lookup',
    ':tensor',
  ],
)

cc_library(
  name = 'memory_mapped_lookup_table',
  srcs = [
    'memory_mapped_lookup_table.hpp',
  ],
  deps = [
    ':blob',
    ':lookup_table',
  ],
)

cc_library(
  name = 'mirrored_index',
  srcs = [
    'mirrored_index.hpp',
  ],
  deps = [
    ':checks',
    ':default_types',
    ':index_strategy',
    ':testing',
  ],
)

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
  name = 'strides',
  srcs = [
    'strides.hpp',
  ],
  deps = [
    ':default_types',
  ],
)

cc_library(
  name = 'tensor',
  srcs = [
    'tensor.hpp',
  ],
  deps = [
    ':checked_index',
    ':checks',
    ':default_types',
    ':functional',
    ':identity_lookup',
    ':index_strategy',
    ':lookup_strategy',
    ':strides',
    ':tensor_interface',
    ':testing',
    ':wrapped_index',
  ],
)

cc_library(
  name = 'tensor_interface',
  srcs = [
    'tensor_interface.hpp',
  ],
  deps = [
    ':checks',
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
    ':checked_index',
    ':default_types',
    ':functional',
    ':identity_lookup',
    ':index_strategy',
    ':lookup_strategy',
    ':tensor_interface',
    ':tensor',
    ':wrapped_index',
  ],
)

cc_library(
  name = 'testing',
  srcs = [
    'testing.hpp',
  ],
)

cc_library(
  name = 'tiled_index_strategy',
  srcs = [
    'tiled_index_strategy.hpp',
  ],
  deps = [
    ':checks',
    ':index_strategy',
  ],
)

cc_library(
  name = 'wrapped_index',
  srcs = [
    'wrapped_index.hpp',
  ],
  deps = [
    ':checks',
    ':default_types',
    ':index_strategy',
  ],
)

cc_test(
  name = 'array_test',
  srcs = [
    'array_test.cpp',
  ],
  size = 'small',
  deps = [
    ':array',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'blob_test',
  srcs = [
    'blob_test.cpp',
  ],
  size = 'small',
  deps = [
    ':array',
    ':blob',
    '//third_party:gtest',
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

cc_test(
  name = 'checks_test',
  srcs = [
    'checks_test.cpp',
  ],
  size = 'small',
  deps = [
    ':checks',
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
  name = 'tensor_test',
  srcs = [
    'tensor_test.cpp',
  ],
  size = 'small',
  deps = [
    ':mirrored_index',
    ':tensor',
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

cc_binary(
  name = 'convcl',
  srcs = [
    'convcl.cpp',
  ],
  linkopts = [
    '-framework OpenCL',
  ],
)

cc_binary(
  name = 'gemmcl',
  srcs = [
    'gemmcl.cpp',
  ],
  linkopts = [
    '-framework OpenCL',
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
