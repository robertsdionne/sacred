
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
  copts = [
    '-I/usr/local/include/',
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
  copts = [
    '-I/usr/local/include/',
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
  copts = [
    '-I/usr/local/include/',
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
  copts = [
    '-I/usr/local/include/',
  ],
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
  copts = [
    '-I/usr/local/include/',
  ],
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
  copts = [
    '-I/usr/local/include/',
  ],
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
  copts = [
    '-I/usr/local/include/',
  ],
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
  copts = [
    '-I/usr/local/include/',
  ],
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
  copts = [
    '-I/usr/local/include/',
  ],
  deps = [
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'tensor_test',
  srcs = [
    'functional.hpp',
    'tensor.hpp',
    'tensor_interface.hpp',
    'tensor_test.cpp',
    'testing.hpp',
  ],
  size = 'small',
  copts = [
    '-I/usr/local/include/',
  ],
  deps = [
    '//third_party:gtest',
  ],
)
