// Star7 SM75 dynamic block-sparse QK-INT8 / PV-FP16 attention core.
// The quantizer is derived from Comfy Kitchen and the FP16-PV kernel from the
// Apache-2.0 SageAttention SM75 implementation; see third_party licenses.

#include <cstdint>

#include "third_party/comfy_kitchen_sage/qk_int_sv_i8_cuda.cuh"
#include "sla_sm75_fp16_kernel.cuh"
#include "sla_sm75_preprocess.cuh"
#include "sol_sm75_preprocess.cuh"
#include "third_party/comfy_kitchen_sage/quant_v_int8.cu"

#include <cuda_runtime.h>

#if defined(_WIN32)
#define STAR7_EXPORT extern "C" __declspec(dllexport)
#else
#define STAR7_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
constexpr int kHeadDim = 128;
constexpr int kCtaQ = 128;
constexpr int kCtaK = 64;
constexpr int kWarpQ = 16;
constexpr int kWarpK = 64;
constexpr int kWarps = (kCtaQ / kWarpQ) * (kCtaK / kWarpK);
constexpr int kShared =
    (kCtaQ + kCtaK) * kHeadDim * sizeof(int8_t) +
    kCtaK * kHeadDim * sizeof(half) + 128;
constexpr int kSolCtaQ = 64;
constexpr int kSolCtaK = 64;
constexpr int kSolWarpQ = 16;
constexpr int kSolWarpK = 64;
constexpr int kSolWarps =
    (kSolCtaQ / kSolWarpQ) * (kSolCtaK / kSolWarpK);
constexpr int kSolShared =
    (kSolCtaQ + kSolCtaK) * kHeadDim * sizeof(int8_t) +
    kSolCtaK * kHeadDim * sizeof(half) + 128;
} // namespace

STAR7_EXPORT int star7_sla_sm75_abi_version() { return 7; }
STAR7_EXPORT int star7_sla_sm75_shared_bytes() { return kShared; }

STAR7_EXPORT int star7_sla_sm75_mean_pool(
    std::uintptr_t input, std::uintptr_t mean, std::uintptr_t output,
    int batch, int heads, int length, int block, int subtract_mean,
    std::uintptr_t stream) {
  if (!input || !output || batch <= 0 || heads <= 0 || length <= 0 ||
      (block != 64 && block != 128) ||
      (subtract_mean != 0 && subtract_mean != 1) ||
      (subtract_mean && !mean)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t error = star7_sla_preprocess::mean_pool(
      reinterpret_cast<const half *>(input),
      reinterpret_cast<const half *>(mean), reinterpret_cast<half *>(output),
      batch, heads, length, block, subtract_mean != 0,
      reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? 0 : 20000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sla_sm75_quantize(
    std::uintptr_t input, std::uintptr_t mean, std::uintptr_t output,
    std::uintptr_t scale, int batch, int heads, int length, int block,
    float multiplier, int subtract_mean, std::uintptr_t stream) {
  if (!input || !output || !scale || batch <= 0 || heads <= 0 || length <= 0 ||
      (block != 16 && block != 64) ||
      (subtract_mean != 0 && subtract_mean != 1) ||
      (subtract_mean && !mean)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t error = star7_sla_preprocess::quantize(
      reinterpret_cast<const half *>(input),
      reinterpret_cast<const half *>(mean), reinterpret_cast<std::int8_t *>(output),
      reinterpret_cast<float *>(scale), batch, heads, length, block, multiplier,
      subtract_mean != 0, reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? 0 : 21000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sla_sm75_launch(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t q_scale, std::uintptr_t k_scale, std::uintptr_t lut,
    std::uintptr_t output, int batch, int heads, int length,
    int selected_blocks, int q_block_base, int q_block_count,
    float attention_scale, std::uintptr_t stream) {
  if (!q || !k || !v || !q_scale || !k_scale || !lut || !output ||
      batch <= 0 || heads <= 0 || length <= 0 || selected_blocks <= 0 ||
      q_block_base < 0 || q_block_count <= 0 ||
      q_block_base + q_block_count > (length + kCtaQ - 1) / kCtaQ) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto kernel = star7_sm75_sparse_qk_i8_pv_f16<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim>;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
  if (error != cudaSuccess) return 1000 + static_cast<int>(error);

  const int stride_h = length * kHeadDim;
  const int stride_b = heads * stride_h;
  dim3 grid(q_block_count, heads, batch);
  dim3 block(32, kWarps);
  kernel<<<grid, block, kShared, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<int8_t *>(q), reinterpret_cast<int8_t *>(k),
      reinterpret_cast<half *>(v), reinterpret_cast<half *>(output),
      reinterpret_cast<float *>(q_scale), reinterpret_cast<float *>(k_scale),
      reinterpret_cast<const int32_t *>(lut), length, selected_blocks,
      q_block_base, stride_b, stride_h, attention_scale, nullptr, 0,
      nullptr, nullptr, nullptr, nullptr, 0, 0,
      nullptr, nullptr, nullptr, nullptr, 0);

  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 2000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sla_sm75_quant_v_int8(
    std::uintptr_t v, std::uintptr_t output, std::uintptr_t scale,
    int batch, int heads, int length, int head_dim, int padded_length,
    std::int64_t stride_b, std::int64_t stride_h, std::int64_t stride_n,
    int dtype_code, std::uintptr_t stream) {
  if (!v || !output || !scale || batch <= 0 || heads <= 0 || length <= 0 ||
      head_dim != kHeadDim || padded_length < length || padded_length % 64 != 0 ||
      (dtype_code != 1 && dtype_code != 2)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  // The vendored quantizer uses 0=float32 and every other supported code as FP16.
  try {
    launch_quant_v_int8_kernel(
        reinterpret_cast<const void *>(v), reinterpret_cast<void *>(output),
        reinterpret_cast<void *>(scale), batch, heads, length, head_dim,
        padded_length, stride_b, stride_h, stride_n, 1,
        reinterpret_cast<cudaStream_t>(stream));
  } catch (...) {
    return 3000;
  }
  const cudaError_t error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 3100 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sla_sm75_launch_all_int8(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t q_scale, std::uintptr_t k_scale, std::uintptr_t v_scale,
    std::uintptr_t lut, std::uintptr_t output, int batch, int heads, int length,
    int padded_length, int selected_blocks, int q_block_base,
    int q_block_count, float attention_scale, std::uintptr_t stream) {
  if (!q || !k || !v || !q_scale || !k_scale || !v_scale || !lut || !output ||
      batch <= 0 || heads <= 0 || length <= 0 || padded_length < length ||
      padded_length % kCtaK != 0 || selected_blocks <= 0 ||
      q_block_base < 0 || q_block_count <= 0 ||
      q_block_base + q_block_count > (length + kCtaQ - 1) / kCtaQ) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto kernel = qk_int_sv_i8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
      QuantGranularity::kPerWarp, QuantGranularity::kPerBlock, float, false,
      half, ComputeUnit::kCudaCore, MaskMode::kNone, false, true, false, false,
      true>;
  constexpr int all_int8_shared = (kCtaQ + kCtaK) * kHeadDim +
      kHeadDim * kCtaK;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, all_int8_shared);
  if (error != cudaSuccess) return 4000 + static_cast<int>(error);

  const std::uint32_t stride_b = heads * length * kHeadDim;
  const std::uint32_t stride_h = length * kHeadDim;
  const std::uint32_t stride_v_b = heads * kHeadDim * padded_length;
  const std::uint32_t stride_v_h = kHeadDim * padded_length;
  dim3 grid(q_block_count, heads, batch);
  dim3 block(32, kWarps);
  kernel<<<grid, block, all_int8_shared,
      reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<int8_t *>(q), reinterpret_cast<int8_t *>(k),
      reinterpret_cast<int8_t *>(v), reinterpret_cast<half *>(output), nullptr,
      reinterpret_cast<float *>(q_scale), reinterpret_cast<float *>(k_scale),
      reinterpret_cast<float *>(v_scale), nullptr, nullptr, 0, 0, 0, 0, 0,
      length, length, 1, stride_b, kHeadDim, stride_h, stride_b, kHeadDim,
      stride_h, stride_v_b, stride_v_h, padded_length, stride_b, kHeadDim,
      stride_h, attention_scale, reinterpret_cast<const int32_t *>(lut),
      selected_blocks, q_block_base, nullptr, 0);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 5000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_shared_bytes() { return kSolShared; }

STAR7_EXPORT int star7_sol_sm75_prepare_routes(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t q_centroid, std::uintptr_t k_centroid,
    std::uintptr_t v_centroid, std::uintptr_t k_mean,
    std::uintptr_t k_variance, std::uintptr_t threshold,
    std::uintptr_t exact_mask, std::uintptr_t row_count,
    int batch, int heads, int length, int padded_blocks,
    float tau, float attention_scale, int sink_start_block,
    int sink_end_block, std::uintptr_t stream) {
  const int blocks = (length + kSolCtaK - 1) / kSolCtaK;
  if (!q || !k || !v || !q_centroid || !k_centroid || !v_centroid ||
      !k_mean || !k_variance || !threshold || !exact_mask || !row_count ||
      batch <= 0 || heads <= 0 || length <= 0 || padded_blocks < blocks ||
      padded_blocks % 16 != 0 || tau < 0.0f || sink_start_block < 0 ||
      sink_end_block < sink_start_block || sink_end_block > blocks) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t error = star7_sol_preprocess::prepare_routes(
      reinterpret_cast<const half *>(q), reinterpret_cast<const half *>(k),
      reinterpret_cast<const half *>(v), reinterpret_cast<half *>(q_centroid),
      reinterpret_cast<half *>(k_centroid),
      reinterpret_cast<half *>(v_centroid), reinterpret_cast<float *>(k_mean),
      reinterpret_cast<float *>(k_variance),
      reinterpret_cast<float *>(threshold),
      reinterpret_cast<std::uint8_t *>(exact_mask),
      reinterpret_cast<std::int32_t *>(row_count), batch, heads, length,
      padded_blocks, tau, attention_scale, sink_start_block, sink_end_block,
      reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_pack_lut(
    std::uintptr_t exact_mask, std::uintptr_t row_count, std::uintptr_t lut,
    int rows, int key_blocks, int lut_stride, int complement,
    std::uintptr_t stream) {
  if (!exact_mask || !row_count || !lut || rows <= 0 || key_blocks <= 0 ||
      lut_stride <= 0 || lut_stride > key_blocks ||
      (complement != 0 && complement != 1)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t error = star7_sol_preprocess::pack_lut(
      reinterpret_cast<const std::uint8_t *>(exact_mask),
      reinterpret_cast<const std::int32_t *>(row_count),
      reinterpret_cast<std::int32_t *>(lut), rows, key_blocks, lut_stride,
      complement != 0, reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? 0 : 11000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_quantize(
    std::uintptr_t input, std::uintptr_t output, std::uintptr_t scale,
    int batch, int heads, int length, int block, std::uintptr_t stream) {
  if (!input || !output || !scale || batch <= 0 || heads <= 0 || length <= 0 ||
      (block != 16 && block != 64)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t error = star7_sol_preprocess::quantize(
      reinterpret_cast<const half *>(input),
      reinterpret_cast<std::int8_t *>(output), reinterpret_cast<float *>(scale),
      batch, heads, length, block, reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? 0 : 12000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_quantize_v_with_scale(
    std::uintptr_t input, std::uintptr_t scale, std::uintptr_t output,
    int batch, int heads, int length, int padded_length,
    std::uintptr_t stream) {
  if (!input || !scale || !output || batch <= 0 || heads <= 0 || length <= 0 ||
      padded_length < length || padded_length % kSolCtaK != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t error = star7_sol_preprocess::quantize_v_with_scale(
      reinterpret_cast<const half *>(input), reinterpret_cast<const float *>(scale),
      reinterpret_cast<std::int8_t *>(output), batch, heads, length,
      padded_length, reinterpret_cast<cudaStream_t>(stream));
  return error == cudaSuccess ? 0 : 13000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_launch(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t q_scale, std::uintptr_t k_scale,
    std::uintptr_t row_count, std::uintptr_t lut,
    std::uintptr_t k_centroid, std::uintptr_t v_centroid,
    std::uintptr_t k_centroid_scale, std::uintptr_t exact_mask,
    std::uintptr_t output, int batch, int heads, int length, int lut_stride,
    int centroid_count, int centroid_padded,
    float attention_scale, std::uintptr_t stream) {
  const int q_blocks = (length + kSolCtaQ - 1) / kSolCtaQ;
  if (!q || !k || !v || !q_scale || !k_scale || !row_count || !lut ||
      !k_centroid || !v_centroid || !k_centroid_scale || !exact_mask ||
      !output || batch <= 0 || heads <= 0 || length <= 0 || lut_stride <= 0 ||
      centroid_count != (length + kSolCtaK - 1) / kSolCtaK ||
      centroid_padded < centroid_count || centroid_padded % kSolCtaK != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto kernel = star7_sm75_sparse_qk_i8_pv_f16<
      kSolCtaQ, kSolCtaK, kSolWarpQ, kSolWarpK, kHeadDim, true>;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSolShared);
  if (error != cudaSuccess) return 6000 + static_cast<int>(error);
  const int stride_h = length * kHeadDim;
  const int stride_b = heads * stride_h;
  dim3 grid(q_blocks, heads, batch);
  dim3 block(32, kSolWarps);
  kernel<<<grid, block, kSolShared, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<int8_t *>(q), reinterpret_cast<int8_t *>(k),
      reinterpret_cast<half *>(v), reinterpret_cast<half *>(output),
      reinterpret_cast<float *>(q_scale), reinterpret_cast<float *>(k_scale),
      reinterpret_cast<const int32_t *>(lut), length, lut_stride, 0,
      stride_b, stride_h, attention_scale,
      reinterpret_cast<const int32_t *>(row_count), lut_stride,
      reinterpret_cast<const int8_t *>(k_centroid),
      reinterpret_cast<const half *>(v_centroid),
      reinterpret_cast<const float *>(k_centroid_scale),
      reinterpret_cast<const uint8_t *>(exact_mask),
      centroid_count, centroid_padded, nullptr, nullptr, nullptr, nullptr, 0);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 7000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_launch_all_int8(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t q_scale, std::uintptr_t k_scale, std::uintptr_t v_scale,
    std::uintptr_t row_count, std::uintptr_t lut, std::uintptr_t output,
    int batch, int heads, int length, int padded_length, int lut_stride,
    float attention_scale, std::uintptr_t stream) {
  const int q_blocks = (length + kSolCtaQ - 1) / kSolCtaQ;
  if (!q || !k || !v || !q_scale || !k_scale || !v_scale || !row_count ||
      !lut || !output || batch <= 0 || heads <= 0 || length <= 0 ||
      padded_length < length || padded_length % kSolCtaK != 0 ||
      lut_stride <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto kernel = qk_int_sv_i8_attn_kernel<
      kSolCtaQ, kSolCtaK, kSolWarpQ, kSolWarpK, kHeadDim, DataType::kInt8,
      QuantGranularity::kPerWarp, QuantGranularity::kPerBlock, float, false,
      half, ComputeUnit::kCudaCore, MaskMode::kNone, false, true, false, false,
      true>;
  constexpr int shared = (kSolCtaQ + kSolCtaK) * kHeadDim +
      kHeadDim * kSolCtaK;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (error != cudaSuccess) return 8000 + static_cast<int>(error);
  const std::uint32_t stride_b = heads * length * kHeadDim;
  const std::uint32_t stride_h = length * kHeadDim;
  const std::uint32_t stride_v_b = heads * kHeadDim * padded_length;
  const std::uint32_t stride_v_h = kHeadDim * padded_length;
  dim3 grid(q_blocks, heads, batch);
  dim3 block(32, kSolWarps);
  kernel<<<grid, block, shared, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<int8_t *>(q), reinterpret_cast<int8_t *>(k),
      reinterpret_cast<int8_t *>(v), reinterpret_cast<half *>(output), nullptr,
      reinterpret_cast<float *>(q_scale), reinterpret_cast<float *>(k_scale),
      reinterpret_cast<float *>(v_scale), nullptr, nullptr, 0, 0, 0, 0, 0,
      length, length, 1, stride_b, kHeadDim, stride_h, stride_b, kHeadDim,
      stride_h, stride_v_b, stride_v_h, padded_length, stride_b, kHeadDim,
      stride_h, attention_scale, reinterpret_cast<const int32_t *>(lut),
      lut_stride, 0, reinterpret_cast<const int32_t *>(row_count), lut_stride);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 9000 + static_cast<int>(error);
}

STAR7_EXPORT int star7_sol_sm75_launch_all_int8_complete(
    std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
    std::uintptr_t q_scale, std::uintptr_t k_scale, std::uintptr_t v_scale,
    std::uintptr_t row_count, std::uintptr_t lut,
    std::uintptr_t k_centroid, std::uintptr_t v_centroid,
    std::uintptr_t k_centroid_scale, std::uintptr_t v_centroid_scale,
    std::uintptr_t exact_mask,
    std::uintptr_t output, int batch, int heads, int length,
    int padded_length, int lut_stride, int centroid_count,
    int centroid_padded, float attention_scale, std::uintptr_t stream) {
  const int q_blocks = (length + kSolCtaQ - 1) / kSolCtaQ;
  if (!q || !k || !v || !q_scale || !k_scale || !v_scale || !row_count ||
      !lut || !k_centroid || !v_centroid || !k_centroid_scale ||
      !v_centroid_scale || !exact_mask ||
      !output || batch <= 0 || heads <= 0 || length <= 0 ||
      padded_length < length || padded_length % kSolCtaK != 0 ||
      lut_stride <= 0 || centroid_count != q_blocks ||
      centroid_padded < centroid_count || centroid_padded % kSolCtaK != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto kernel = star7_sm75_sparse_qk_i8_pv_f16<
      kSolCtaQ, kSolCtaK, kSolWarpQ, kSolWarpK, kHeadDim, true, true>;
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSolShared);
  if (error != cudaSuccess) return 14000 + static_cast<int>(error);
  const int stride_h = length * kHeadDim;
  const int stride_b = heads * stride_h;
  dim3 grid(q_blocks, heads, batch);
  dim3 block(32, kSolWarps);
  kernel<<<grid, block, kSolShared, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<int8_t *>(q), reinterpret_cast<int8_t *>(k), nullptr,
      reinterpret_cast<half *>(output), reinterpret_cast<float *>(q_scale),
      reinterpret_cast<float *>(k_scale), reinterpret_cast<const int32_t *>(lut),
      length, lut_stride, 0, stride_b, stride_h, attention_scale,
      reinterpret_cast<const int32_t *>(row_count), lut_stride,
      reinterpret_cast<const int8_t *>(k_centroid), nullptr,
      reinterpret_cast<const float *>(k_centroid_scale),
      reinterpret_cast<const uint8_t *>(exact_mask), centroid_count,
      centroid_padded, reinterpret_cast<const int8_t *>(v),
      reinterpret_cast<const int8_t *>(v_centroid),
      reinterpret_cast<const float *>(v_scale),
      reinterpret_cast<const float *>(v_centroid_scale), padded_length);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 15000 + static_cast<int>(error);
}
