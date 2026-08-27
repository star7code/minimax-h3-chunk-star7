#pragma once

#include "third_party/comfy_kitchen_sage/cp_async.cuh"
#include "third_party/comfy_kitchen_sage/math.cuh"
#include "third_party/comfy_kitchen_sage/mma.cuh"
#include "third_party/comfy_kitchen_sage/permuted_smem.cuh"
#include "third_party/comfy_kitchen_sage/attn_utils.cuh"

__device__ __forceinline__ void star7_mma_m16n8k8_f16_f32(
    float *accumulator, const uint32_t *a, const uint32_t *b) {
  float result[4];
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};\n"
      : "=&f"(result[0]), "=&f"(result[1]), "=&f"(result[2]), "=&f"(result[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]),
        "f"(accumulator[0]), "f"(accumulator[1]),
        "f"(accumulator[2]), "f"(accumulator[3]));
#pragma unroll
  for (int i = 0; i < 4; ++i) accumulator[i] = result[i];
}

__device__ __forceinline__ void star7_ldmatrix_m8n8x1_trans(
    uint32_t *fragment, half *pointer) {
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x1.shared.b16 {%0}, [%1];\n"
      : "=r"(fragment[0]) : "r"(address));
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_K,
          uint32_t TILES_Q, uint32_t TILES_K, uint32_t TILES_V,
          uint32_t HEAD_DIM>
__device__ __forceinline__ void star7_compute_fp16_sv_sm75(
    half *smem_v, half *smem_probability,
    float probability[][TILES_K][8], float output[][TILES_V][8]) {
  const uint32_t lane = get_lane_id();
  const uint32_t warp = get_warp_id();
  half *warp_probability = smem_probability + warp * 16 * 16;
#pragma unroll
  for (uint32_t fk = 0; fk < TILES_K; ++fk) {
#pragma unroll
    for (uint32_t fq = 0; fq < TILES_Q; ++fq) {
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t row = fq * 16 + 8 * ((i % 4) / 2) + lane / 4;
        const uint32_t col = 8 * (i / 4) + i % 2 + 2 * (lane % 4);
        warp_probability[row * 16 + col] = __float2half_rn(probability[fq][fk][i]);
      }
    }
    __syncwarp();
#pragma unroll
    for (uint32_t nk = 0; nk < 2; ++nk) {
      uint32_t p_fragment[2];
      mma::ldmatrix_m8n8x2(
          p_fragment, warp_probability + (lane % 16) * 16 + nk * 8);
#pragma unroll
      for (uint32_t fv = 0; fv < TILES_V; ++fv) {
#pragma unroll
        for (uint32_t out_half = 0; out_half < 2; ++out_half) {
          uint32_t v_fragment[1];
          half *v_ptr = smem_v +
              (fk * 16 + nk * 8 + lane % 8) * HEAD_DIM +
              fv * 16 + out_half * 8;
          star7_ldmatrix_m8n8x1_trans(v_fragment, v_ptr);
#pragma unroll
          for (uint32_t fq = 0; fq < TILES_Q; ++fq)
            star7_mma_m16n8k8_f16_f32(
                output[fq][fv] + out_half * 4, p_fragment, v_fragment);
        }
      }
    }
    __syncwarp();
  }
}

template <uint32_t CTA_K, uint32_t HEAD_DIM, uint32_t NUM_WARPS,
          SwizzleMode SWIZZLE>
__device__ __forceinline__ void star7_load_int8_v_sm75(
    const int8_t *__restrict__ source, uint32_t padded_length,
    uint32_t key_start, const smem_t<SWIZZLE, CTA_K / 16> &smem_v) {
  const uint32_t thread = get_warp_id() * 32 + get_lane_id();
  constexpr uint32_t vectors_per_row = CTA_K / 16;
  constexpr uint32_t vectors = HEAD_DIM * vectors_per_row;
  for (uint32_t vector = thread; vector < vectors;
       vector += NUM_WARPS * 32) {
    const uint32_t dim = vector / vectors_per_row;
    const uint32_t token_vector = vector % vectors_per_row;
    const uint32_t offset = smem_v.get_permuted_offset(dim, token_vector);
    smem_v.load_128b_async(
        offset, source + dim * padded_length + key_start + token_vector * 16);
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_K,
          uint32_t TILES_Q, uint32_t TILES_K, uint32_t TILES_V,
          SwizzleMode SWIZZLE, uint32_t STRIDE>
__device__ __forceinline__ void star7_compute_int8_sv_scaled_sm75(
    const smem_t<SWIZZLE, STRIDE> &smem_v,
    int32_t probability_scale_bits[][TILES_K][8],
    uint32_t probability_u8[][TILES_K / 2][4],
    float output[][TILES_V][8], const float *__restrict__ value_scale) {
  const uint32_t lane = get_lane_id();
  uint32_t smem_v_row = lane % 8 + (lane / 16) * 8;
  const uint32_t smem_v_col = (lane / 8) % 2;
  uint32_t offsets[TILES_K / 2];
#pragma unroll
  for (uint32_t fk = 0; fk < TILES_K / 2; ++fk)
    offsets[fk] = smem_v.get_permuted_offset(smem_v_row, smem_v_col + fk * 2);
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; ++fv) {
    uint32_t values[TILES_K / 2][4];
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K / 2; ++fk) {
      smem_v.ldmatrix_m8n8x4(offsets[fk], values[fk]);
      offsets[fk] = smem_v.template advance_offset_by_row<16>(offsets[fk]);
    }
    float channel_scale[4];
    const float *scale_base = value_scale + fv * 16 + (lane % 4) * 2;
    reinterpret_cast<float2 *>(channel_scale)[0] =
        *reinterpret_cast<const float2 *>(scale_base);
    reinterpret_cast<float2 *>(channel_scale)[1] =
        *reinterpret_cast<const float2 *>(scale_base + 8);
#pragma unroll
    for (uint32_t fq = 0; fq < TILES_Q; ++fq) {
      int32_t partial[8];
      mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInit>(
          partial, probability_u8[fq][0], values[0]);
#pragma unroll
      for (uint32_t fk = 1; fk < TILES_K / 2; ++fk)
        mma::mma_sync_m16n16k32_row_col_u8s8s32<
            mma::MMAMode::kInplaceUpdate>(
            partial, probability_u8[fq][fk], values[fk]);
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) {
        const float probability_scale =
            __int_as_float(probability_scale_bits[fq][0][(i % 4) / 2]);
        const float scale = channel_scale[(i / 4) * 2 + (i % 2)];
        output[fq][fv][i] = fmaf(
            __int2float_rn(partial[i]), probability_scale * scale,
            output[fq][fv][i]);
      }
    }
  }
}

template <uint32_t NUM_WARPS_Q, uint32_t NUM_WARPS_K,
          uint32_t TILES_Q, uint32_t TILES_K, uint32_t TILES_V,
          SwizzleMode SWIZZLE, uint32_t STRIDE>
__device__ __forceinline__ void star7_compute_int8_pv_sm75(
    float probabilities[][TILES_K][8], float output[][TILES_V][8],
    float row_max[][2], float row_sum[][2], float score_scale,
    int32_t score_scratch[][TILES_K][8],
    const smem_t<SWIZZLE, STRIDE> &smem_v,
    const float *__restrict__ value_scale) {
  float probability_scale[TILES_Q][2];
  update_mdo<TILES_Q, TILES_K, TILES_V, true, false>(
      probabilities, output, row_max, row_sum, probability_scale,
      score_scale, S_U8_OFFSET);
  uint32_t probability_u8[TILES_Q][TILES_K / 2][4];
  RS_to_u8<TILES_Q, TILES_K>(probabilities, probability_u8);
  accumulate_d<TILES_Q, TILES_K>(
      probabilities, row_sum, probability_scale);
#pragma unroll
  for (uint32_t fq = 0; fq < TILES_Q; ++fq)
#pragma unroll
    for (uint32_t k = 0; k < 2; ++k)
      score_scratch[fq][0][k] = __float_as_int(probability_scale[fq][k]);
  star7_compute_int8_sv_scaled_sm75<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q,
      TILES_K, TILES_V, SWIZZLE, STRIDE>(
      smem_v, score_scratch, probability_u8, output, value_scale);
}

template <uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q,
          uint32_t WARP_K, uint32_t HEAD_DIM,
          bool ENABLE_CENTROID_APPROX = false, bool PV_INT8 = false>
__global__ void star7_sm75_sparse_qk_i8_pv_f16(
    const int8_t *__restrict__ Q, const int8_t *__restrict__ K,
    const half *__restrict__ V, half *__restrict__ O,
    const float *__restrict__ QScale, const float *__restrict__ KScale,
    const int32_t *__restrict__ Lut, uint32_t length,
    uint32_t selected_blocks, uint32_t q_block_base,
    uint32_t stride_b, uint32_t stride_h,
    float attention_scale, const int32_t *__restrict__ RowCount = nullptr,
    uint32_t lut_stride = 0,
    const int8_t *__restrict__ KCentroid = nullptr,
    const half *__restrict__ VCentroid = nullptr,
    const float *__restrict__ KCentroidScale = nullptr,
    const uint8_t *__restrict__ ExactMask = nullptr,
    uint32_t centroid_count = 0, uint32_t centroid_padded = 0,
    const int8_t *__restrict__ VInt8 = nullptr,
    const int8_t *__restrict__ VCentroidInt8 = nullptr,
    const float *__restrict__ VScale = nullptr,
    const float *__restrict__ VCentroidScale = nullptr,
    uint32_t v_padded_length = 0) {
  constexpr uint32_t PACK_QK = 16;
  constexpr uint32_t MMA_M = 16;
  constexpr uint32_t MMA_N = 16;
  constexpr uint32_t NUM_WARPS_Q = CTA_Q / WARP_Q;
  constexpr uint32_t NUM_WARPS_K = CTA_K / WARP_K;
  constexpr uint32_t NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_K;
  constexpr uint32_t TILES_Q = WARP_Q / MMA_M;
  constexpr uint32_t TILES_K = WARP_K / MMA_N;
  constexpr uint32_t TILES_V = HEAD_DIM / MMA_N;
  constexpr uint32_t INNER = HEAD_DIM / 32;
  constexpr SwizzleMode QK_SWIZZLE = SwizzleMode::k128B;
  constexpr uint32_t QK_LINE_LANES = 8;
  constexpr uint32_t QK_COPY_LINES = 4;
  constexpr uint32_t Q_ROW_ITERS = HEAD_DIM / (QK_LINE_LANES * PACK_QK);
  constexpr uint32_t Q_COL_ITERS = CTA_Q / (NUM_WARPS * QK_COPY_LINES);
  constexpr uint32_t K_COL_ITERS = CTA_K / (NUM_WARPS * QK_COPY_LINES);

  extern __shared__ int8_t shared[];
  smem_t<QK_SWIZZLE, HEAD_DIM / PACK_QK> smem_q(shared);
  smem_t<QK_SWIZZLE, HEAD_DIM / PACK_QK> smem_k(
      shared + CTA_Q * HEAD_DIM);
  half *smem_v = reinterpret_cast<half *>(
      shared + (CTA_Q + CTA_K) * HEAD_DIM);
  constexpr SwizzleMode V_INT8_SWIZZLE = SwizzleMode::k64B;
  smem_t<V_INT8_SWIZZLE, CTA_K / 16> smem_v_int8(
      shared + (CTA_Q + CTA_K) * HEAD_DIM);

  const uint32_t lane = get_lane_id();
  const uint32_t warp = get_warp_id();
  const uint32_t batch = blockIdx.z;
  const uint32_t head = blockIdx.y;
  const uint32_t local_q_block = blockIdx.x;
  const uint32_t q_block = q_block_base + local_q_block;
  const uint32_t heads = gridDim.y;
  const uint32_t q_blocks = div_ceil(length, CTA_Q);
  const uint64_t route_row =
      (static_cast<uint64_t>(batch) * heads + head) * q_blocks + q_block;
  const uint32_t active_selected_blocks = RowCount
      ? static_cast<uint32_t>(RowCount[route_row]) : selected_blocks;
  const uint64_t lut_base = RowCount
      ? route_row * lut_stride
      : ((static_cast<uint64_t>(batch) * heads + head) * gridDim.x + local_q_block) *
            selected_blocks;

  int32_t scores[TILES_Q][TILES_K][8];
  float output[TILES_Q][TILES_V][8];
  float row_max[TILES_Q][2];
  float row_sum[TILES_Q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < TILES_Q; ++fq) {
#pragma unroll
    for (uint32_t fv = 0; fv < TILES_V; ++fv)
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) output[fq][fv][i] = 0.0f;
    row_max[fq][0] = row_max[fq][1] = -50000.0f;
    row_sum[fq][0] = row_sum[fq][1] = 0.0f;
  }

  const uint32_t q_start = q_block * CTA_Q;
  int8_t *q_ptr = const_cast<int8_t *>(Q) + batch * stride_b + head * stride_h +
      (q_start + CTA_Q / NUM_WARPS * warp + lane / QK_LINE_LANES) * HEAD_DIM +
      (lane % QK_LINE_LANES) * PACK_QK;
  uint32_t q_smem_load = smem_q.get_permuted_offset(
      warp * QK_COPY_LINES * Q_COL_ITERS + lane / QK_LINE_LANES,
      lane % QK_LINE_LANES);
  uint32_t q_load_row = q_start + CTA_Q / NUM_WARPS * warp +
      lane / QK_LINE_LANES;
  load_global_to_share<QK_LINE_LANES, QK_COPY_LINES, Q_ROW_ITERS,
      Q_COL_ITERS, QK_SWIZZLE, HEAD_DIM / PACK_QK, CTA_Q>(
      &q_ptr, q_smem_load, HEAD_DIM, smem_q, q_load_row, length);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  const uint32_t q_mma_base = smem_q.get_permuted_offset(
      get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_K>() * WARP_Q + lane % 16,
      lane / 16);
  const uint32_t q_warp = get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_K>();
  const float q_scale = QScale[
      ((batch * heads + head) * q_blocks + q_block) * NUM_WARPS_Q + q_warp];
  const uint32_t key_blocks = div_ceil(length, CTA_K);

  for (uint32_t selected = 0; selected < active_selected_blocks; ++selected) {
    const uint32_t key_block = static_cast<uint32_t>(Lut[lut_base + selected]);
    const uint32_t key_start = key_block * CTA_K;
    int8_t *k_ptr = const_cast<int8_t *>(K) + batch * stride_b + head * stride_h +
        (key_start + CTA_K / NUM_WARPS * warp + lane / QK_LINE_LANES) * HEAD_DIM +
        (lane % QK_LINE_LANES) * PACK_QK;
    uint32_t k_smem_load = smem_k.get_permuted_offset(
        warp * QK_COPY_LINES * K_COL_ITERS + lane / QK_LINE_LANES,
        lane % QK_LINE_LANES);
    const uint32_t k_load_row = key_start + CTA_K / NUM_WARPS * warp +
        lane / QK_LINE_LANES;
    load_global_to_share<QK_LINE_LANES, QK_COPY_LINES, Q_ROW_ITERS,
        K_COL_ITERS, QK_SWIZZLE, HEAD_DIM / PACK_QK, CTA_K>(
        &k_ptr, k_smem_load, HEAD_DIM, smem_k, k_load_row, length);
    if constexpr (PV_INT8) {
      const int8_t *v_head = VInt8 +
          (static_cast<uint64_t>(batch) * heads + head) *
              HEAD_DIM * v_padded_length;
      star7_load_int8_v_sm75<CTA_K, HEAD_DIM, NUM_WARPS, V_INT8_SWIZZLE>(
          v_head, v_padded_length, key_start, smem_v_int8);
    } else {
      for (uint32_t vector = warp * 32 + lane; vector < CTA_K * HEAD_DIM / 8;
           vector += NUM_WARPS * 32) {
        const uint32_t row = vector / (HEAD_DIM / 8);
        const uint32_t column = (vector % (HEAD_DIM / 8)) * 8;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (key_start + row < length)
          value = *reinterpret_cast<const uint4 *>(
              V + batch * stride_b + head * stride_h +
              (key_start + row) * HEAD_DIM + column);
        *reinterpret_cast<uint4 *>(smem_v + row * HEAD_DIM + column) = value;
      }
    }
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    uint32_t q_mma = q_mma_base;
    uint32_t k_mma = smem_k.get_permuted_offset(
        get_warp_idx_k<NUM_WARPS_Q, NUM_WARPS_K>() * WARP_K + lane % 8 +
            (lane / 16) * 8,
        (lane / 8) % 2);
    compute_int_qk<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q, TILES_K, INNER,
        QK_SWIZZLE, HEAD_DIM / PACK_QK, DataType::kInt8>(
        smem_q, smem_k, scores, q_mma, k_mma);

    float probabilities[TILES_Q][TILES_K][8];
#pragma unroll
    for (uint32_t fq = 0; fq < TILES_Q; ++fq)
#pragma unroll
      for (uint32_t fk = 0; fk < TILES_K; ++fk)
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i)
          probabilities[fq][fk][i] = __int2float_rz(scores[fq][fk][i]);
    const uint32_t k_lane_base = key_start +
        get_warp_idx_k<NUM_WARPS_Q, NUM_WARPS_K>() * WARP_K +
        2 * (lane % 4);
    if (key_block == key_blocks - 1)
      apply_out_of_bound_mask<TILES_Q, TILES_K>(
          k_lane_base, probabilities, length, -1.0e30f);

    const float score_scale = attention_scale * math::log2e * q_scale *
        KScale[(batch * heads + head) * key_blocks + key_block];
    if constexpr (PV_INT8) {
      star7_compute_int8_pv_sm75<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q,
          TILES_K, TILES_V, V_INT8_SWIZZLE, CTA_K / 16>(
          probabilities, output, row_max, row_sum, score_scale, scores,
          smem_v_int8,
          VScale + (static_cast<uint64_t>(batch) * heads + head) * HEAD_DIM);
    } else {
      float probability_scale[TILES_Q][2];
      update_mdo<TILES_Q, TILES_K, TILES_V, false, false>(
          probabilities, output, row_max, row_sum, probability_scale,
          score_scale, 0.0f);
      accumulate_d<TILES_Q, TILES_K>(probabilities, row_sum, probability_scale);
#pragma unroll
      for (uint32_t fq = 0; fq < TILES_Q; ++fq)
#pragma unroll
        for (uint32_t fk = 0; fk < TILES_K; ++fk)
#pragma unroll
          for (uint32_t i = 0; i < 8; ++i)
            probabilities[fq][fk][i] *= probability_scale[fq][(i % 4) / 2];
      __syncthreads();
      half *smem_probability = reinterpret_cast<half *>(
          shared + CTA_Q * HEAD_DIM);
      star7_compute_fp16_sv_sm75<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q, TILES_K,
          TILES_V, HEAD_DIM>(smem_v, smem_probability, probabilities, output);
    }
    __syncthreads();
  }

  if constexpr (ENABLE_CENTROID_APPROX) {
    const uint32_t centroid_groups = centroid_padded / CTA_K;
    const uint32_t centroid_stride_h = centroid_padded * HEAD_DIM;
    const uint32_t centroid_stride_b = heads * centroid_stride_h;
    for (uint32_t key_block = 0; key_block < centroid_groups; ++key_block) {
      const uint32_t key_start = key_block * CTA_K;
      int8_t *k_ptr = const_cast<int8_t *>(KCentroid) +
          batch * centroid_stride_b + head * centroid_stride_h +
          (key_start + CTA_K / NUM_WARPS * warp + lane / QK_LINE_LANES) * HEAD_DIM +
          (lane % QK_LINE_LANES) * PACK_QK;
      uint32_t k_smem_load = smem_k.get_permuted_offset(
          warp * QK_COPY_LINES * K_COL_ITERS + lane / QK_LINE_LANES,
          lane % QK_LINE_LANES);
      const uint32_t k_load_row = key_start + CTA_K / NUM_WARPS * warp +
          lane / QK_LINE_LANES;
      load_global_to_share<QK_LINE_LANES, QK_COPY_LINES, Q_ROW_ITERS,
          K_COL_ITERS, QK_SWIZZLE, HEAD_DIM / PACK_QK, CTA_K>(
          &k_ptr, k_smem_load, HEAD_DIM, smem_k, k_load_row, centroid_count);
      if constexpr (PV_INT8) {
        const int8_t *v_centroid_head = VCentroidInt8 +
            (static_cast<uint64_t>(batch) * heads + head) *
                HEAD_DIM * centroid_padded;
        star7_load_int8_v_sm75<CTA_K, HEAD_DIM, NUM_WARPS, V_INT8_SWIZZLE>(
            v_centroid_head, centroid_padded, key_start, smem_v_int8);
      } else {
        for (uint32_t vector = warp * 32 + lane; vector < CTA_K * HEAD_DIM / 8;
             vector += NUM_WARPS * 32) {
          const uint32_t row = vector / (HEAD_DIM / 8);
          const uint32_t column = (vector % (HEAD_DIM / 8)) * 8;
          uint4 value = make_uint4(0, 0, 0, 0);
          if (key_start + row < centroid_count)
            value = *reinterpret_cast<const uint4 *>(
                VCentroid + batch * centroid_stride_b + head * centroid_stride_h +
                (key_start + row) * HEAD_DIM + column);
          *reinterpret_cast<uint4 *>(smem_v + row * HEAD_DIM + column) = value;
        }
      }
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();

      uint32_t q_mma = q_mma_base;
      uint32_t k_mma = smem_k.get_permuted_offset(
          get_warp_idx_k<NUM_WARPS_Q, NUM_WARPS_K>() * WARP_K + lane % 8 +
              (lane / 16) * 8,
          (lane / 8) % 2);
      compute_int_qk<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q, TILES_K, INNER,
          QK_SWIZZLE, HEAD_DIM / PACK_QK, DataType::kInt8>(
          smem_q, smem_k, scores, q_mma, k_mma);

      float probabilities[TILES_Q][TILES_K][8];
#pragma unroll
      for (uint32_t fq = 0; fq < TILES_Q; ++fq)
#pragma unroll
        for (uint32_t fk = 0; fk < TILES_K; ++fk)
#pragma unroll
          for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t k_lane_base = key_start +
                get_warp_idx_k<NUM_WARPS_Q, NUM_WARPS_K>() * WARP_K +
                2 * (lane % 4);
            const uint32_t centroid = k_lane_base + fk * 16 +
                8 * (i / 4) + i % 2;
            const bool valid_centroid = centroid < centroid_count;
            const bool selected_exact = valid_centroid &&
                ExactMask[route_row * centroid_count + centroid] != 0;
            const uint32_t block_start = centroid * CTA_K;
            const uint32_t block_length = valid_centroid
                ? min(CTA_K, length - block_start) : 1;
            probabilities[fq][fk][i] =
                (valid_centroid && !selected_exact)
                ? fmaf(__int2float_rz(scores[fq][fk][i]),
                       attention_scale * math::log2e * q_scale *
                           KCentroidScale[(batch * heads + head) *
                               centroid_groups + key_block],
                       math::ptx_log2(static_cast<float>(block_length)))
                : -1.0e30f;
          }
      if constexpr (PV_INT8) {
        star7_compute_int8_pv_sm75<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q,
            TILES_K, TILES_V, V_INT8_SWIZZLE, CTA_K / 16>(
            probabilities, output, row_max, row_sum, 1.0f, scores,
            smem_v_int8, VCentroidScale +
                (static_cast<uint64_t>(batch) * heads + head) * HEAD_DIM);
      } else {
        float probability_scale[TILES_Q][2];
        update_mdo<TILES_Q, TILES_K, TILES_V, false, false>(
            probabilities, output, row_max, row_sum, probability_scale,
            1.0f, 0.0f);
        accumulate_d<TILES_Q, TILES_K>(probabilities, row_sum, probability_scale);
#pragma unroll
        for (uint32_t fq = 0; fq < TILES_Q; ++fq)
#pragma unroll
          for (uint32_t fk = 0; fk < TILES_K; ++fk)
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i)
              probabilities[fq][fk][i] *= probability_scale[fq][(i % 4) / 2];
        __syncthreads();
        half *smem_probability = reinterpret_cast<half *>(
            shared + CTA_Q * HEAD_DIM);
        star7_compute_fp16_sv_sm75<NUM_WARPS_Q, NUM_WARPS_K, TILES_Q, TILES_K,
            TILES_V, HEAD_DIM>(smem_v, smem_probability, probabilities, output);
      }
      __syncthreads();
    }
  }

  row_sum[0][0] += __shfl_xor_sync(0xffffffff, row_sum[0][0], 1);
  row_sum[0][0] += __shfl_xor_sync(0xffffffff, row_sum[0][0], 2);
  row_sum[0][1] += __shfl_xor_sync(0xffffffff, row_sum[0][1], 1);
  row_sum[0][1] += __shfl_xor_sync(0xffffffff, row_sum[0][1], 2);

  // SM75 m16n8k8 holds rows r and r+8 in each lane group, not adjacent
  // rows.  The public SM75 fork used 2r/2r+1 here, which silently permuted
  // seven eighths of the output rows while still passing uniform-QK tests.
  const uint32_t output_row0 = lane / 4;
  const uint32_t output_row1 = output_row0 + 8;
  const float denominator0 = row_sum[0][0];
  const float denominator1 = row_sum[0][1];
  const float inverse0 = denominator0 > 0.0f ? math::ptx_rcp(denominator0) : 0.0f;
  const float inverse1 = denominator1 > 0.0f ? math::ptx_rcp(denominator1) : 0.0f;

  half *smem_o = reinterpret_cast<half *>(shared);
  const uint32_t warp_row =
      get_warp_idx_q<NUM_WARPS_Q, NUM_WARPS_K>() * WARP_Q;
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; ++fv) {
#pragma unroll
    for (uint32_t out_half = 0; out_half < 2; ++out_half) {
      const uint32_t column = fv * 16 + out_half * 8 + (lane % 4) * 2;
      float *fragment = output[0][fv] + out_half * 4;
      smem_o[(warp_row + output_row0) * HEAD_DIM + column] =
          __float2half_rn(fragment[0] * inverse0);
      smem_o[(warp_row + output_row0) * HEAD_DIM + column + 1] =
          __float2half_rn(fragment[1] * inverse0);
      smem_o[(warp_row + output_row1) * HEAD_DIM + column] =
          __float2half_rn(fragment[2] * inverse1);
      smem_o[(warp_row + output_row1) * HEAD_DIM + column + 1] =
          __float2half_rn(fragment[3] * inverse1);
    }
  }
  __syncthreads();

  for (uint32_t index = warp * 32 + lane; index < CTA_Q * HEAD_DIM;
       index += NUM_WARPS * 32) {
    const uint32_t row = index / HEAD_DIM;
    if (q_start + row < length)
      O[batch * stride_b + head * stride_h +
        (q_start + row) * HEAD_DIM + index % HEAD_DIM] = smem_o[index];
  }
}
