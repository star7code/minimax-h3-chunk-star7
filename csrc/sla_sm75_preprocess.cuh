#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// Small, dependency-free preprocessing kernels for the SM75 SLA path.
// They deliberately mirror sla_backend.py's bounded PyTorch implementation:
// FP32 reductions, optional K-mean subtraction, per-block symmetric INT8
// scaling, and half-away-from-zero rounding.
namespace star7_sla_preprocess {

constexpr int kHeadDim = 128;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

__global__ void mean_pool_kernel(
    const half *__restrict__ input, const half *__restrict__ mean,
    half *__restrict__ output, int batch_heads, int length, int block, int groups,
    bool subtract_mean) {
  const std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const std::int64_t total =
      static_cast<std::int64_t>(batch_heads) * groups * kHeadDim;
  if (linear >= total) return;
  std::int64_t value = linear;
  const int dim = static_cast<int>(value % kHeadDim);
  value /= kHeadDim;
  const int group = static_cast<int>(value % groups);
  const int bh = static_cast<int>(value / groups);
  const int start = group * block;
  const int count = min(block, length - start);
  const std::int64_t base = static_cast<std::int64_t>(bh) * length * kHeadDim;
  const float center = subtract_mean
      ? __half2float(mean[static_cast<std::int64_t>(bh) * kHeadDim + dim])
      : 0.0f;
  float sum = 0.0f;
  for (int token = 0; token < count; ++token) {
    sum += __half2float(input[
        base + static_cast<std::int64_t>(start + token) * kHeadDim + dim]) - center;
  }
  output[linear] = __float2half_rn(sum / static_cast<float>(count));
}

template <int TOKENS>
__global__ void quantize_kernel(
    const half *__restrict__ input, const half *__restrict__ mean,
    std::int8_t *__restrict__ output, float *__restrict__ scales,
    int length, int groups, float multiplier, bool subtract_mean) {
  __shared__ float warp_maxima[8];
  const int linear = static_cast<int>(blockIdx.x);
  const int bh = linear / groups;
  const int group = linear - bh * groups;
  const int start = group * TOKENS;
  const std::int64_t base = static_cast<std::int64_t>(bh) * length * kHeadDim;
  float maximum = 0.0f;
  for (int item = static_cast<int>(threadIdx.x); item < TOKENS * kHeadDim;
       item += static_cast<int>(blockDim.x)) {
    const int token = item / kHeadDim;
    const int dim = item - token * kHeadDim;
    if (start + token < length) {
      const float center = subtract_mean
          ? __half2float(mean[static_cast<std::int64_t>(bh) * kHeadDim + dim])
          : 0.0f;
      const float source = (__half2float(input[
          base + static_cast<std::int64_t>(start + token) * kHeadDim + dim]) -
          center) * multiplier;
      maximum = fmaxf(maximum, fabsf(source));
    }
  }
  maximum = warp_max(maximum);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) warp_maxima[warp] = maximum;
  __syncthreads();
  if (warp == 0) {
    maximum = lane < 8 ? warp_maxima[lane] : 0.0f;
    maximum = warp_max(maximum);
    if (lane == 0) warp_maxima[0] = maximum;
  }
  __syncthreads();
  const float scale = fmaxf(warp_maxima[0] / 127.0f, 1.0e-8f);
  if (threadIdx.x == 0)
    scales[static_cast<std::int64_t>(bh) * groups + group] = scale;
  for (int item = static_cast<int>(threadIdx.x); item < TOKENS * kHeadDim;
       item += static_cast<int>(blockDim.x)) {
    const int token = item / kHeadDim;
    const int dim = item - token * kHeadDim;
    if (start + token < length) {
      const std::int64_t index = base +
          static_cast<std::int64_t>(start + token) * kHeadDim + dim;
      const float center = subtract_mean
          ? __half2float(mean[static_cast<std::int64_t>(bh) * kHeadDim + dim])
          : 0.0f;
      const float normalized =
          ((__half2float(input[index]) - center) * multiplier) / scale;
      const int rounded = static_cast<int>(truncf(
          normalized + (normalized >= 0.0f ? 0.5f : -0.5f)));
      output[index] = static_cast<std::int8_t>(max(-127, min(127, rounded)));
    }
  }
}

inline cudaError_t mean_pool(
    const half *input, const half *mean, half *output, int batch, int heads,
    int length, int block, bool subtract_mean, cudaStream_t stream) {
  const int groups = (length + block - 1) / block;
  const std::int64_t total =
      static_cast<std::int64_t>(batch) * heads * groups * kHeadDim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mean_pool_kernel<<<blocks, threads, 0, stream>>>(
      input, mean, output, batch * heads, length, block, groups, subtract_mean);
  return cudaPeekAtLastError();
}

inline cudaError_t quantize(
    const half *input, const half *mean, std::int8_t *output, float *scale,
    int batch, int heads, int length, int block, float multiplier,
    bool subtract_mean, cudaStream_t stream) {
  const int groups = (length + block - 1) / block;
  const int grid = batch * heads * groups;
  if (block == 16) {
    quantize_kernel<16><<<grid, 256, 0, stream>>>(
        input, mean, output, scale, length, groups, multiplier, subtract_mean);
  } else {
    quantize_kernel<64><<<grid, 256, 0, stream>>>(
        input, mean, output, scale, length, groups, multiplier, subtract_mean);
  }
  return cudaPeekAtLastError();
}

} // namespace star7_sla_preprocess
