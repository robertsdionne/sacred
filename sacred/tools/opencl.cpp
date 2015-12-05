#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>
#include <iostream>
#include <string>
#include <vector>

#include "opencl/cl.hpp"
#include "sacred/proto/opencl.pb.h"

int main(int argument_count, char *arguments[]) {
  using std::cout;
  using std::endl;
  using std::string;
  using std::vector;

  auto opencl = sacred::proto::OpenCl();

  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  for (auto &platform : platforms) {
    auto proto_platform = opencl.add_platform();
    proto_platform->set_profile(platform.getInfo<CL_PLATFORM_PROFILE>());
    proto_platform->set_version(platform.getInfo<CL_PLATFORM_VERSION>());
    proto_platform->set_name(platform.getInfo<CL_PLATFORM_NAME>());
    proto_platform->set_vendor(platform.getInfo<CL_PLATFORM_VENDOR>());
    proto_platform->set_extensions(platform.getInfo<CL_PLATFORM_EXTENSIONS>());
  }

  auto proto_platform = opencl.mutable_platform(0);
  auto platform = platforms.at(0);

  vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  for (auto &device : devices) {
    auto proto_device = proto_platform->add_device();
    proto_device->set_type(static_cast<sacred::proto::Device_Type>(device.getInfo<CL_DEVICE_TYPE>()));
    proto_device->set_vendor_identifier(device.getInfo<CL_DEVICE_VENDOR_ID>());
    proto_device->set_maximum_compute_units(device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    proto_device->set_maximum_work_item_dimensions(device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>());
    for (auto size : device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()) {
      proto_device->add_maximum_work_item_sizes(size);
    }
    proto_device->set_maximum_work_group_size(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    proto_device->set_preferred_vector_width_character(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>());
    proto_device->set_preferred_vector_width_short(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>());
    proto_device->set_preferred_vector_width_integer(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>());
    proto_device->set_preferred_vector_width_long(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>());
    proto_device->set_preferred_vector_width_float(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>());
    proto_device->set_preferred_vector_width_double(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>());
    proto_device->set_preferred_vector_width_half(device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>());
    proto_device->set_native_vector_width_character(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>());
    proto_device->set_native_vector_width_short(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>());
    proto_device->set_native_vector_width_integer(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>());
    proto_device->set_native_vector_width_long(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>());
    proto_device->set_native_vector_width_float(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>());
    proto_device->set_native_vector_width_double(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>());
    proto_device->set_native_vector_width_half(device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>());
    proto_device->set_address_bits(device.getInfo<CL_DEVICE_ADDRESS_BITS>());
    proto_device->set_maximum_memory_allocation_size(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    proto_device->set_image_support(device.getInfo<CL_DEVICE_IMAGE_SUPPORT>());
    proto_device->set_maximum_read_image_arguments(device.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>());
    proto_device->set_maximum_write_image_arguments(device.getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>());
    proto_device->set_image2d_maximum_width(device.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>());
    proto_device->set_image2d_maximum_height(device.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>());
    proto_device->set_image3d_maximum_width(device.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>());
    proto_device->set_image3d_maximum_height(device.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>());
    proto_device->set_image3d_maximum_depth(device.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>());
    proto_device->set_maximum_samplers(device.getInfo<CL_DEVICE_MAX_SAMPLERS>());
    proto_device->set_maximum_parameter_size(device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>());
    proto_device->set_memory_base_address_alignment(device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>());
    proto_device->set_minimum_data_type_alignment_size(device.getInfo<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE>());
    for (auto i = 0; i < 9; ++i) {
      if (device.getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() & (1 << i)) {
        proto_device->add_single_floating_point_configuration(
            static_cast<sacred::proto::Device_FloatingPointConfiguration>(1 << i));
      }
    }
    proto_device->set_global_memory_cache_type(
        static_cast<sacred::proto::Device_MemoryCacheType>(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>()));
    proto_device->set_global_memory_cacheline_size(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>());
    proto_device->set_global_memory_cache_size(device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>());
    proto_device->set_global_memory_size(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());
    proto_device->set_maximum_constant_buffer_size(device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>());
    proto_device->set_maximum_constant_arguments(device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>());
    proto_device->set_local_memory_type(
        static_cast<sacred::proto::Device_LocalMemoryType>(device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>()));
    proto_device->set_local_memory_size(device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    proto_device->set_error_correction_support(device.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>());
    proto_device->set_host_unified_memory(device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>());
    proto_device->set_profiling_timer_resolution(device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>());
    proto_device->set_endian_little(device.getInfo<CL_DEVICE_ENDIAN_LITTLE>());
    proto_device->set_available(device.getInfo<CL_DEVICE_AVAILABLE>());
    proto_device->set_compiler_available(device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>());
    for (auto i = 0; i < 3; ++i) {
      if (device.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>() & (1 << i)) {
        proto_device->add_execution_capabilities(
            static_cast<sacred::proto::Device_ExecutionCapability>(1 << i));
      }
    }
    for (auto i = 0; i < 3; ++i) {
      if (device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>() & (1 << i)) {
        proto_device->add_queue_properties(
            static_cast<sacred::proto::Device_QueueProperty>(1 << i));
      }
    }
    proto_device->set_name(device.getInfo<CL_DEVICE_NAME>());
    proto_device->set_vendor(device.getInfo<CL_DEVICE_VENDOR>());
    proto_device->set_driver_version(device.getInfo<CL_DRIVER_VERSION>());
    proto_device->set_profile(device.getInfo<CL_DEVICE_PROFILE>());
    proto_device->set_version(device.getInfo<CL_DEVICE_VERSION>());
    proto_device->set_opencl_c_version(device.getInfo<CL_DEVICE_OPENCL_C_VERSION>());
    proto_device->set_extensions(device.getInfo<CL_DEVICE_EXTENSIONS>());
  }

  string json;
  auto type_resolver = google::protobuf::util::NewTypeResolverForDescriptorPool(
      "sacred", google::protobuf::DescriptorPool::generated_pool());

  auto json_options = google::protobuf::util::JsonOptions();
  json_options.add_whitespace = true;
  google::protobuf::util::BinaryToJsonString(
      type_resolver, "sacred/" + opencl.GetDescriptor()->full_name(), opencl.SerializeAsString(), &json, json_options);

  cout << json << endl;

  return 0;
}
