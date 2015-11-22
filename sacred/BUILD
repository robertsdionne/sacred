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

cc_library(
  name = 'rectifier_gradient',
  srcs = [
    'rectifier_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
    '//external:googlelog',
  ],
)

cc_test(
  name = 'rectifier_gradient_test',
  srcs = [
    'rectifier_gradient_test.cpp',
  ],
  size = 'small',
  deps = [
    ':rectifier_gradient',
    ':tensor',
    '//third_party:gtest',
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
