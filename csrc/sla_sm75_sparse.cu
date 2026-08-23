// Star7 SM75 dynamic block-sparse QK-INT8 / PV-FP16 attention core.
// The quantizer is derived from Comfy Kitchen and the FP16-PV kernel from the
// Apache-2.0 SageAttention SM75 implementation; see third_party licenses.

#include "sla_sm75_fp16_kernel.cuh"
#include "third_party/comfy_kitchen_sage/qk_int_sv_i8_cuda.cuh"
#include "third_party/comfy_kitchen_sage/quant_v_int8.cu"

#include <cstdint>
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
} // namespace

STAR7_EXPORT int star7_sla_sm75_abi_version() { return 7; }
STAR7_EXPORT int star7_sla_sm75_shared_bytes() { return kShared; }

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
      q_block_base, stride_b, stride_h, attention_scale);

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
      selected_blocks, q_block_base);
  error = cudaPeekAtLastError();
  return error == cudaSuccess ? 0 : 5000 + static_cast<int>(error);
}
