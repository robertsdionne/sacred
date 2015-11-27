package(
  default_visibility = ['//visibility:public'],
)

cc_library(
  name = 'convolution',
  srcs = [
    'convolution.hpp',
  ],
  deps = [
    ':default_types',
    ':math',
    ':operator',
  ],
)

cc_library(
  name = 'convolution_gradient',
  srcs = [
    'convolution_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
  ],
)

cc_test(
  name = 'convolution_gradient_test',
  srcs = [
    'convolution_gradient_test.cpp',
  ],
  size = 'small',
  deps = [
    ':convolution',
    ':convolution_gradient',
    ':dual',
    ':gradients',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'convolution_test',
  srcs = [
    'convolution_test.cpp',
  ],
  size = 'small',
  deps = [
    ':convolution',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'default_types',
  srcs = [
    'default_types.hpp',
  ],
  deps = [
    ':dual',
  ],
)

cc_library(
  name = 'dual',
  srcs = [
    'dual.hpp',
  ],
)

cc_test(
  name = 'dual_test',
  srcs = [
    'dual_test.cpp',
  ],
  size = 'small',
  deps = [
    ':dual',
    '//third_party:gtest',
  ],
)

cc_library(
  name = 'exponential_linear',
  srcs = [
    'exponential_linear.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
    '//external:googlelog',
  ],
)

cc_library(
  name = 'exponential_linear_gradient',
  srcs = [
    'exponential_linear_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
    '//external:googlelog',
  ],
)

cc_test(
  name = 'exponential_linear_gradient_test',
  srcs = [
    'exponential_linear_gradient_test.cpp',
  ],
  size = 'small',
  deps = [
    ':dual',
    ':gradients',
    ':exponential_linear',
    ':exponential_linear_gradient',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'exponential_linear_test',
  srcs = [
    'exponential_linear_test.cpp',
  ],
  size = 'small',
  deps = [
    ':exponential_linear',
    ':tensor',
    '//third_party:gtest',
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
    ':dual',
    ':gradients',
    ':fully_connected',
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
  name = 'gradients',
  srcs = [
    'gradients.hpp',
  ],
  deps = [
    ':default_types',
    ':dual',
    ':tensor',
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
    ':dual',
    ':gradients',
    ':rectifier',
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
  name = 'softmax',
  srcs = [
    'softmax.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
  ],
)

cc_library(
  name = 'softmax_gradient',
  srcs = [
    'softmax_gradient.hpp',
  ],
  deps = [
    ':default_types',
    ':operator',
  ],
)

cc_test(
  name = 'softmax_gradient_test',
  srcs = [
    'softmax_gradient_test.cpp',
  ],
  size = 'small',
  deps = [
    ':dual',
    ':gradients',
    ':softmax',
    ':softmax_gradient',
    ':tensor',
    '//third_party:gtest',
  ],
)

cc_test(
  name = 'softmax_test',
  srcs = [
    'softmax_test.cpp',
  ],
  size = 'small',
  deps = [
    ':softmax',
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
