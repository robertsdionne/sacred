#include <OpenCL/OpenCL.h>
#include <cassert>
#include <iostream>

/**
 * Multiplies matrices A and B to produce the result C.
 *
 * In Einstein notation: C^i_j = beta * C^i_j + alpha * A^{ik} * B_{kj}
 */
const char *kKernelSource = u8R"openclc(

kernel void ReConv2(
    global T *c, global const T *a, const int j,
    const int K, const int L, const int M, const int N,
    const T alpha, const T beta) {
  int i = get_global_id(0);
  T c_out = beta * c[i * N + j];
  for (int k = 0; k < K; ++k) {
    for (int l = 0; l < L; ++l) {
      int x = j + l - L;
      int y = i + k - K / 2;
      bool in = 0 <= x && 0 <= y && y < M;
      if (in) {
        c_out += alpha * a[k * L + l] * c[y * N + x];
      }
    }
  }
  c[i * N + j] = c_out;
}

kernel void Conv2(
    global T *c, global const T *a, global const T *b,
    const int K, const int L, const int M, const int N,
    const T alpha, const T beta) {
  int i = get_global_id(0), j = get_global_id(1);
  T c_out = beta * c[i * (N - L + 1) + j];
  for (int k = 0; k < K; ++k) {
    for (int l = 0; l < L; ++l) {
      c_out += alpha * a[k * L + l] * b[(i + k) * N + (j + l)];
    }
  }
  c[i * (N - L + 1) + j] = c_out;
}

kernel void GemmNN(
    global T *c, global const T *a, global const T *b,
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
    global T *c, global const T *a, global const T *b,
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
void Print(const float *a) {
  for (auto i = 0; i < M; ++i) {
    for (auto j = 0; j < N; ++j) {
      std::cout << a[i * N + j] << u8" ";
    }
    std::cout << std::endl;
  }
}

int main(int argument_count, char *arguments[]) {
  constexpr auto K = 3, L = 3, M = 4, N = 4;
  float c[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };
  float a[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  float b[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  // float *C = new float[(M - K + 1) * (N - L + 1)], *A = new float[K * L], *B = new float[M * N];
  //
  // for (auto i = 0; i < (M - K + 1) * (N - L + 1); ++i) {
  //   C[i] = 0;
  // }
  //
  // for (auto i = 0; i < K * L; ++i) {
  //   A[i] = i;
  // }
  //
  // for (auto i = 0; i < M * N; ++i) {
  //   B[i] = i;
  // }

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

  size_t max_work_items[3];
  assert(CL_SUCCESS == clGetDeviceInfo(
      device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_items), max_work_items, nullptr));
  // std::cout << max_work_items[0] << std::endl;
  // std::cout << max_work_items[1] << std::endl;
  // std::cout << max_work_items[2] << std::endl;

  cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0
  };
  cl_int error;
  cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
  assert(CL_SUCCESS == error);

  cl_mem buffer_a = clCreateBuffer(
      context, CL_MEM_READ_ONLY, K * L * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_mem buffer_b = clCreateBuffer(
      context, CL_MEM_READ_ONLY, M * N * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);
  cl_mem buffer_c = clCreateBuffer(
      context, CL_MEM_READ_WRITE, M * N * sizeof(float), nullptr, &error);
  assert(CL_SUCCESS == error);

  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_a, CL_TRUE, 0, K * L * sizeof(float), a, 0, nullptr, nullptr));
  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_b, CL_TRUE, 0, M * N * sizeof(float), b, 0, nullptr, nullptr));
  assert(CL_SUCCESS == clEnqueueWriteBuffer(
      queue, buffer_c, CL_TRUE, 0, M * N * sizeof(float), c, 0, nullptr, nullptr));

  cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &error);
  assert(CL_SUCCESS == error);

  std::string options = u8"-DT=float";
  assert(CL_SUCCESS == clBuildProgram(program, 1, &device, options.c_str(), nullptr, nullptr));
  assert(CL_SUCCESS == clUnloadPlatformCompiler(platform));

  cl_kernel kernel = clCreateKernel(program, u8"ReConv2", &error);
  assert(CL_SUCCESS == error);

  float alpha = 1.0, beta = 1.0;
  assert(CL_SUCCESS == clSetKernelArg(kernel, 0, sizeof(buffer_c), &buffer_c));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 1, sizeof(buffer_a), &buffer_a));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 3, sizeof(int), &K));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 4, sizeof(int), &L));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 5, sizeof(int), &M));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 6, sizeof(int), &N));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 7, sizeof(float), &alpha));
  assert(CL_SUCCESS == clSetKernelArg(kernel, 8, sizeof(float), &beta));

  const size_t global_work_size[] = {M};
  for (int j = 0; j < N; ++j) {
    assert(CL_SUCCESS == clSetKernelArg(kernel, 2, sizeof(int), &j));
    assert(CL_SUCCESS == clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));
  }

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
