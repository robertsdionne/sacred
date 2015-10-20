
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

cc_binary(
  name = 'sacred',
  srcs = [
    'sacred.cpp',
  ],
  deps = [
    '//external:googleflags',
  ],
)

cc_test(
  name = 'array_test',
  srcs = [
    'array_test.cpp',
  ],
  size = 'small',
  deps = [
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
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'math_test',
  srcs = [
    'math_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'nonrecursive_filter_layer_test',
  srcs = [
    'nonrecursive_filter_layer_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'reconv_test',
  srcs = [
    'reconv_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'recursive_filter_layer_test',
  srcs = [
    'recursive_filter_layer_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'slice_test',
  srcs = [
    'slice.hpp',
    'slice_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'stacked_test',
  srcs = [
    'stacked_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'strides_test',
  srcs = [
    'strides.hpp',
    'strides_test.cpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'tensor_test',
  srcs = [
    'functional.hpp',
    'checked_index.hpp',
    'hashed_lookup.hpp',
    'identity_index.hpp',
    'identity_lookup.hpp',
    'index_strategy.hpp',
    'lookup_strategy.hpp',
    'mirrored_index.hpp',
    'strides.hpp',
    'tensor.hpp',
    'tensor_interface.hpp',
    'tensor_test.cpp',
    'testing.hpp',
    'wrapped_index.hpp',
  ],
  size = 'small',
  deps = [
    '//third_party:gtest',
  ],
)
