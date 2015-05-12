#include <OpenCL/OpenCL.h>
#include <cassert>
#include <iostream>

/**
 * Multiplies matrices A and B to produce the result C.
 *
 * In Einstein notation: C^i_j = beta * C^i_j + alpha * A^{ik} * B_{kj}
 */
const char *kKernelSource = u8R"openclc(

kernel void GemmNN(
    global T *c, constant T *a, constant T *b,
    const int M, const int N, const int K,
    const T alpha, const T beta) {
  int i = get_global_id(0), j = get_global_id(1);
  T c_out = beta * c[i * N + j];
  for (int k = 0; k < K; ++k) {
    c_out += alpha * a[i * K + k] * b[k * N + j];
  }
  c[i * N + j] = c_out;
}

kernel void GemmNT(
    global T *c, constant T *a, constant T *b,
    const int M, const int N, const int K,
    const T alpha, const T beta) {
  int i = get_global_id(0), j = get_global_id(1);
  T c_out = beta * c[i * N + j];
  for (int k = 0; k < K; ++k) {
    c_out += alpha * a[i * K + k] * b[j * K + k];
  }
  c[i * N + j] = c_out;
}

)openclc";

template <int M, int N>
void Print(const float a[M][N]) {
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      std::cout << a[i][j] << u8" ";
    }
    std::cout << std::endl;
  }
}

int main(int argument_count, char *arguments[]) {
  constexpr auto M = 4, N = 3, K = 2;
  float c[M][N] = {0};
  float a[M][K] = {
    {1, 2},
    {3, 4},
    {5, 6},
    {7, 8}
  };
  float b[K][N] = {
    {1, 2, 3},
    {4, 5, 6}
  };
  float b_t[N][K] = {
    {1, 4},
    {2, 5},
    {3, 6}
  };

  cl_platform_id platform = 0;
  assert(CL_SUCCESS == clGetPlatformIDs(1, &platform, nullptr));
  cl_device_id device = 0;
  cl_device_id devices[10] = {};
  cl_uint number_of_devices;
  assert(CL_SUCCESS == clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_GPU, 10, devices, &number_of_devices));

  std::string target_gpu = u8"GeForce";

  char buffer[4 * 1024] = {};
  size_t size = 0;
  for (auto i = 0; i < number_of_devices; ++i) {
    assert(
        CL_SUCCESS == clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, &size));
    std::string device_name = buffer;
    if (std::string::npos != device_name.find(target_gpu)) {
      device = devices[i];
    }
  }
  assert(device > 0);

  assert(
      CL_SUCCESS == clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, &size));
  // std::cout << buffer << std::endl;

  cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0
  };
  cl_int error;
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
  assert(CL_SUCCESS == error);

  cl_mem buffer_a = clCreateBuffer(
      context, CL_MEM_READ_ONLY, M * K * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_mem buffer_b = clCreateBuffer(
      context, CL_MEM_READ_ONLY, K * N * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_mem buffer_c = clCreateBuffer(
      context, CL_MEM_READ_WRITE, M * N * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);

  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_a, CL_TRUE, 0, M * K * sizeof(float), a, 0, nullptr, nullptr));
  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_b, CL_TRUE, 0, K * N * sizeof(float), b, 0, nullptr, nullptr));
  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_c, CL_TRUE, 0, M * N * sizeof(float), c, 0, nullptr, nullptr));

  cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &error);
  assert(CL_SUCCESS == error);

  std::string options = u8"-DT=float";
  assert(CL_SUCCESS == clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr));
  assert(CL_SUCCESS == clUnloadPlatformCompiler(platform));

  cl_kernel kernel = clCreateKernel(program, u8"GemmNN", &error);
  assert(CL_SUCCESS == error);

  float alpha = 1.0, beta = 0.0;
  assert(CL_SUCCESS == clSetKernelArg(kernel, 0, sizeof(buffer_c), &buffer_c));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 1, sizeof(buffer_a), &buffer_a));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 2, sizeof(buffer_b), &buffer_b));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 3, sizeof(int), &M));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 4, sizeof(int), &N));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 5, sizeof(int), &K));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 6, sizeof(float), &alpha));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 7, sizeof(float), &beta));

  const size_t global_work_size[] = {M, N};
  assert(CL_SUCCESS == clEnqueueNDRangeKernel(
      queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  assert(CL_SUCCESS == clFinish(queue));

  // TODO(robertsdionne): kernel stuff here

  assert(CL_SUCCESS == clEnqueueReadBuffer(
      queue, buffer_c, CL_TRUE, 0, M * N * sizeof(float), c, 0, nullptr, nullptr));

  Print<M, N>(c);

  assert(CL_SUCCESS == clReleaseMemObject(buffer_a));
  assert(CL_SUCCESS == clReleaseMemObject(buffer_b));
  assert(CL_SUCCESS == clReleaseMemObject(buffer_c));

  assert(CL_SUCCESS == clReleaseKernel(kernel));
  assert(CL_SUCCESS == clReleaseProgram(program));

  assert(CL_SUCCESS == clReleaseCommandQueue(queue));
  assert(CL_SUCCESS == clReleaseContext(context));

  return 0;
}
