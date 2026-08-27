#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

namespace star7_sol_preprocess {

constexpr int kBlock = 64;
constexpr int kHeadDim = 128;
constexpr float kLog2E = 1.4426950408889634f;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

__global__ void reduce_qkv_centroids(
    const half *__restrict__ q, const half *__restrict__ k,
    const half *__restrict__ v, half *__restrict__ qc,
    half *__restrict__ kc, half *__restrict__ vc, int heads, int length,
    int blocks, int padded_blocks) {
  const int linear = static_cast<int>(blockIdx.x);
  const int bh = linear / padded_blocks;
  const int block = linear - bh * padded_blocks;
  const int dim = static_cast<int>(threadIdx.x);
  if (dim >= kHeadDim) return;
  const std::int64_t summary_index =
      (static_cast<std::int64_t>(bh) * padded_blocks + block) * kHeadDim + dim;
  if (block >= blocks) {
    qc[summary_index] = __float2half(0.0f);
    kc[summary_index] = __float2half(0.0f);
    vc[summary_index] = __float2half(0.0f);
    return;
  }
  const int start = block * kBlock;
  const int count = min(kBlock, length - start);
  const std::int64_t base =
      static_cast<std::int64_t>(bh) * length * kHeadDim;
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  float v_sum = 0.0f;
#pragma unroll
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const std::int64_t index = base +
          static_cast<std::int64_t>(start + token) * kHeadDim + dim;
      q_sum += __half2float(q[index]);
      k_sum += __half2float(k[index]);
      v_sum += __half2float(v[index]);
    }
  }
  const float inverse = 1.0f / static_cast<float>(count);
  qc[summary_index] = __float2half_rn(q_sum * inverse);
  kc[summary_index] = __float2half_rn(k_sum * inverse);
  // Store the mean.  The attention kernel adds log2(block_length), which is
  // mathematically identical to NVIDIA Sol's summed V centroid.
  vc[summary_index] = __float2half_rn(v_sum * inverse);
}

__global__ void reduce_kc_stats(
    const half *__restrict__ kc, float *__restrict__ mean,
    float *__restrict__ variance, int blocks, int padded_blocks) {
  const int bh = static_cast<int>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  if (dim >= kHeadDim) return;
  float sum = 0.0f;
  float square_sum = 0.0f;
  const std::int64_t base =
      static_cast<std::int64_t>(bh) * padded_blocks * kHeadDim;
  for (int block = 0; block < blocks; ++block) {
    const float value = __half2float(kc[base + block * kHeadDim + dim]);
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  const float inverse = 1.0f / static_cast<float>(blocks);
  const float average = sum * inverse;
  mean[static_cast<std::int64_t>(bh) * kHeadDim + dim] = average;
  variance[static_cast<std::int64_t>(bh) * kHeadDim + dim] =
      fmaxf(square_sum * inverse - average * average, 0.0f);
}

__global__ void compute_diag_threshold(
    const half *__restrict__ qc, const float *__restrict__ kc_mean,
    const float *__restrict__ kc_variance, float *__restrict__ threshold,
    int q_blocks, int padded_blocks, float tau, float scale_log2) {
  __shared__ float mean_parts[kHeadDim];
  __shared__ float variance_parts[kHeadDim];
  const int route = static_cast<int>(blockIdx.x);
  const int bh = route / q_blocks;
  const int q_block = route - bh * q_blocks;
  const int dim = static_cast<int>(threadIdx.x);
  const std::int64_t centroid_index =
      (static_cast<std::int64_t>(bh) * padded_blocks + q_block) * kHeadDim + dim;
  const std::int64_t stat_index =
      static_cast<std::int64_t>(bh) * kHeadDim + dim;
  const float q_value = __half2float(qc[centroid_index]);
  mean_parts[dim] = q_value * kc_mean[stat_index];
  variance_parts[dim] = q_value * q_value * kc_variance[stat_index];
  __syncthreads();
  for (int stride = kHeadDim / 2; stride > 0; stride >>= 1) {
    if (dim < stride) {
      mean_parts[dim] += mean_parts[dim + stride];
      variance_parts[dim] += variance_parts[dim + stride];
    }
    __syncthreads();
  }
  if (dim == 0) {
    const float mean = mean_parts[0] * scale_log2;
    const float variance = variance_parts[0] * scale_log2 * scale_log2;
    threshold[route] = mean + tau * sqrtf(fmaxf(variance, 0.0f) + 1.0e-6f);
  }
}

__global__ void route_centroid_tiles(
    const half *__restrict__ qc, const half *__restrict__ kc,
    const float *__restrict__ threshold, std::uint8_t *__restrict__ exact_mask,
    std::int32_t *__restrict__ row_count, int heads, int q_blocks,
    int k_blocks, int padded_blocks, float scale_log2,
    int sink_start_block, int sink_end_block) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ float score_tile[16 * 16];
  const int k_tile = static_cast<int>(blockIdx.x);
  const int q_tile = static_cast<int>(blockIdx.y);
  const int bh = static_cast<int>(blockIdx.z);
  const half *a = qc +
      (static_cast<std::int64_t>(bh) * padded_blocks + q_tile * 16) * kHeadDim;
  // Row-major [16,128] is the same byte layout as col-major [128,16].
  const half *b = kc +
      (static_cast<std::int64_t>(bh) * padded_blocks + k_tile * 16) * kHeadDim;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  wmma::fill_fragment(cf, 0.0f);
#pragma unroll
  for (int dim = 0; dim < kHeadDim; dim += 16) {
    wmma::load_matrix_sync(af, a + dim, kHeadDim);
    wmma::load_matrix_sync(bf, b + dim, kHeadDim);
    wmma::mma_sync(cf, af, bf, cf);
  }
  wmma::store_matrix_sync(score_tile, cf, 16, wmma::mem_row_major);
  __syncthreads();
  for (int item = static_cast<int>(threadIdx.x); item < 256; item += 32) {
    const int local_q = item / 16;
    const int local_k = item - local_q * 16;
    const int q_block = q_tile * 16 + local_q;
    const int k_block = k_tile * 16 + local_k;
    if (q_block < q_blocks && k_block < k_blocks) {
      const std::int64_t row = static_cast<std::int64_t>(bh) * q_blocks + q_block;
      const bool neighbor = abs(q_block - k_block) <= 1;
      const bool sink = k_block >= sink_start_block && k_block < sink_end_block;
      const bool selected = score_tile[item] * scale_log2 > threshold[row] ||
          neighbor || sink;
      exact_mask[row * k_blocks + k_block] = selected ? 1 : 0;
      if (selected) atomicAdd(row_count + row, 1);
    }
  }
#endif
}

__global__ void pack_mask_lut(
    const std::uint8_t *__restrict__ exact_mask,
    const std::int32_t *__restrict__ expected_count,
    std::int32_t *__restrict__ lut, int rows, int k_blocks, int lut_stride,
    bool complement) {
  __shared__ int cursor;
  if (threadIdx.x == 0) cursor = 0;
  __syncthreads();
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  for (int base = 0; base < k_blocks; base += static_cast<int>(blockDim.x)) {
    const int key = base + static_cast<int>(threadIdx.x);
    const bool exact = key < k_blocks && exact_mask[
        static_cast<std::int64_t>(row) * k_blocks + key] != 0;
    const bool selected = key < k_blocks && (complement ? !exact : exact);
    const unsigned ballot = __ballot_sync(0xffffffff, selected);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int count = __popc(ballot);
    int offset = 0;
    if (lane == 0) offset = atomicAdd(&cursor, count);
    offset = __shfl_sync(0xffffffff, offset, 0);
    if (selected) {
      const int rank = __popc(ballot & ((1u << lane) - 1u));
      const int slot = offset + rank;
      if (slot < lut_stride)
        lut[static_cast<std::int64_t>(row) * lut_stride + slot] = key;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && !complement && cursor != expected_count[row]) {
    // A mismatch indicates an internal packing error.  Poison the count so the
    // Python ABI validation stops before launching attention.
    const_cast<std::int32_t *>(expected_count)[row] = -1;
  }
}

template <int TOKENS>
__global__ void quantize_blocks(
    const half *__restrict__ input, std::int8_t *__restrict__ output,
    float *__restrict__ scales, int length, int groups) {
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
    if (start + token < length)
      maximum = fmaxf(maximum, fabsf(__half2float(
          input[base + static_cast<std::int64_t>(start + token) * kHeadDim + dim])));
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
      const int value = __float2int_rn(__half2float(input[index]) / scale);
      output[index] = static_cast<std::int8_t>(max(-127, min(127, value)));
    }
  }
}

__global__ void quantize_v_with_scale_kernel(
    const half *__restrict__ input, const float *__restrict__ scale,
    std::int8_t *__restrict__ output, int batch, int heads, int length,
    int padded_length) {
  const std::int64_t total = static_cast<std::int64_t>(batch) * heads *
      kHeadDim * padded_length;
  for (std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       linear < total; linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    std::int64_t value = linear;
    const int token = static_cast<int>(value % padded_length);
    value /= padded_length;
    const int dim = static_cast<int>(value % kHeadDim);
    const int bh = static_cast<int>(value / kHeadDim);
    std::int8_t quantized = 0;
    if (token < length) {
      const float step = fmaxf(
          scale[static_cast<std::int64_t>(bh) * kHeadDim + dim], 1.0e-8f);
      const float source = __half2float(input[
          (static_cast<std::int64_t>(bh) * length + token) * kHeadDim + dim]);
      const int rounded = __float2int_rn(source / step);
      quantized = static_cast<std::int8_t>(max(-127, min(127, rounded)));
    }
    output[(static_cast<std::int64_t>(bh) * kHeadDim + dim) * padded_length + token] =
        quantized;
  }
}

inline cudaError_t prepare_routes(
    const half *q, const half *k, const half *v, half *qc, half *kc, half *vc,
    float *kc_mean, float *kc_variance, float *threshold,
    std::uint8_t *exact_mask, std::int32_t *row_count, int batch, int heads,
    int length, int padded_blocks, float tau, float attention_scale,
    int sink_start_block, int sink_end_block, cudaStream_t stream) {
  const int blocks = (length + kBlock - 1) / kBlock;
  const int bh = batch * heads;
  cudaError_t error = cudaMemsetAsync(
      row_count, 0, static_cast<std::size_t>(bh) * blocks * sizeof(std::int32_t),
      stream);
  if (error != cudaSuccess) return error;
  reduce_qkv_centroids<<<bh * padded_blocks, kHeadDim, 0, stream>>>(
      q, k, v, qc, kc, vc, heads, length, blocks, padded_blocks);
  reduce_kc_stats<<<bh, kHeadDim, 0, stream>>>(
      kc, kc_mean, kc_variance, blocks, padded_blocks);
  const float scale_log2 = attention_scale * kLog2E;
  compute_diag_threshold<<<bh * blocks, kHeadDim, 0, stream>>>(
      qc, kc_mean, kc_variance, threshold, blocks, padded_blocks, tau,
      scale_log2);
  dim3 route_grid((padded_blocks + 15) / 16, (padded_blocks + 15) / 16, bh);
  route_centroid_tiles<<<route_grid, 32, 0, stream>>>(
      qc, kc, threshold, exact_mask, row_count, heads, blocks, blocks,
      padded_blocks, scale_log2, sink_start_block, sink_end_block);
  return cudaPeekAtLastError();
}

inline cudaError_t pack_lut(
    const std::uint8_t *exact_mask, const std::int32_t *row_count,
    std::int32_t *lut, int rows, int k_blocks, int lut_stride,
    bool complement, cudaStream_t stream) {
  pack_mask_lut<<<rows, 256, 0, stream>>>(
      exact_mask, row_count, lut, rows, k_blocks, lut_stride, complement);
  return cudaPeekAtLastError();
}

inline cudaError_t quantize(
    const half *input, std::int8_t *output, float *scale, int batch,
    int heads, int length, int block, cudaStream_t stream) {
  const int groups = (length + block - 1) / block;
  if (block == 16) {
    quantize_blocks<16><<<batch * heads * groups, 256, 0, stream>>>(
        input, output, scale, length, groups);
  } else if (block == 64) {
    quantize_blocks<64><<<batch * heads * groups, 256, 0, stream>>>(
        input, output, scale, length, groups);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

inline cudaError_t quantize_v_with_scale(
    const half *input, const float *scale, std::int8_t *output, int batch,
    int heads, int length, int padded_length, cudaStream_t stream) {
  const std::int64_t total = static_cast<std::int64_t>(batch) * heads *
      kHeadDim * padded_length;
  const std::int64_t requested = (total + 255) / 256;
  const int blocks = static_cast<int>(requested < 65535 ? requested : 65535);
  quantize_v_with_scale_kernel<<<blocks, 256, 0, stream>>>(
      input, scale, output, batch, heads, length, padded_length);
  return cudaPeekAtLastError();
}

}  // namespace star7_sol_preprocess
