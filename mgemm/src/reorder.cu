#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "reorder.cuh"
#include "cutlass/numeric_conversion.h"

#include <cstdio>


#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__

#define FP4_MAX 6
#define FP6_MAX 28
#define FP8_MAX 448

typedef cutlass::float_e2m1_t fp4_t;
typedef cutlass::float_e3m2_t fp6_t;
typedef cutlass::float_e4m3_t fp8_t;
typedef cutlass::float_ue8m0_t sf_t;
typedef cutlass::bfloat16_t bf16_t;

namespace cg = cooperative_groups;
using namespace cute;

struct PackFp4 {
  int8_t low : 4;
  int8_t high : 4;
};

// HOST_DEVICE int cdiv(int a, int b) { return (a + b - 1) / b; }

HOST_DEVICE float fpmax(float a, float b) { return (a) > (b) ? (a) : (b); }

HOST_DEVICE float fpmin(float a, float b) { return (a) < (b) ? (a) : (b); }

HOST_DEVICE float clamp(float x, float a, float b) { return fpmax(a, fpmin(b, x)); }

template <typename T> HOST_DEVICE T abs(T x) { return x < (T)0 ? -x : x; }

template <typename T, typename U, typename Accum, int Size = sizeof(U) / sizeof(T)>
HOST_DEVICE Accum local_sum_p2(U *vec, Accum sumv) {
  T *view = reinterpret_cast<T *>(vec);
  #pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    sumv += (Accum)view[i] * (Accum)view[i];
  }
  return sumv;
}
HOST_DEVICE void pack_4_fp6_to_3_bytes(
    uint8_t fp6_v0, uint8_t fp6_v1, uint8_t fp6_v2, uint8_t fp6_v3,
    uint8_t* output_bytes // Array of 3 uint8_t
) {
    fp6_v0 &= 0x3F; fp6_v1 &= 0x3F; fp6_v2 &= 0x3F; fp6_v3 &= 0x3F;

    output_bytes[0] = (fp6_v0) | ((fp6_v1 & 0x03) << 6);
    output_bytes[1] = (fp6_v1 >> 2) | ((fp6_v2 & 0x0F) << 4);
    output_bytes[2] = (fp6_v2 >> 4) | (fp6_v3 << 2);
}
/*
 * Given a row index, return the start index of the scale.
*/
// HOST_DEVICE int scale_index(int row_id){
//   int bottomUpper = (row_id / 8) % 2;
//   int group_idx = row_id % 8;
//   int group_nums = row_id / 16;
//   return (group_nums * 64) + (group_idx * 8) + bottomUpper;
// }

/*
 * Given the row numbers, calculate the leading dimension of scales.
 * In unit of half.
*/
#define SCALE_SIZE(x) ((x) / 32)
#define GROUP_NUM(x) ((x) / 32)

#define mymax(a, b) ((a) > (b) ? (a) : (b))

template <typename T, typename U, int Size = sizeof(U) / sizeof(T)>
DEVICE float local_abs_max(U *vec, float maxv) {
  T *view = reinterpret_cast<T *>(vec);
  #pragma unroll 4
  for (int i = 0; i < Size; ++i) {
    maxv = mymax((float)maxv, (float)abs((float)view[i]));
  }
  return maxv;
}


// Linear notation:
//   X: [S, D], W: [H, D], Y = X @ W^T: [S, H] => PyTorch layout, D가 matmul K-dim
//   D = in_features = HIDDEN_DIM = KN + KS + KO
//   H = out_features
//   S = activation row 수. QLinear에서는 batch_size * sequence_length.
//
// 이 kernel의 input은 activation X [S, D] 또는 weight W [H, D]이다.
// 두 경우를 함께 표현하기 위해 아래에서는 input을 [R, D]로 표기한다.
//   activation이면 R = S, row_id는 token row
//   weight이면     R = H, row_id는 output-channel row
//
// Mixed-precision output의 physical shape(uint8 기준):
//   f4out: [R, KN / 2]       -- FP4 두 개를 1 byte에 packing
//   f6out: [R, KS * 3 / 4]   -- FP6 4/3개를 1 bytes에 packing
//   f8out: [R, KO]           -- FP8 한 개가 1 byte
// row load, reorder, scale 계산, quantization, packing을 한 kernel에서 수행하므로
// reordered BF16 intermediate [R, D]를 global memory에 만들지 않고 Shared memory에서 처리
// GROUP_SIZE = 32는 한 스레드가 처리하는 개수
template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void reorder_quantize_mixed_kernel(
  bf16_t *input,          // logical shape: [R, D]
  int16_t *reorder_index,// shape: [D], reordered channel -> original channel
  uint8_t *f4out,        // physical shape: [R, KN / 2]
  uint8_t *f6out,        // physical shape: [R, KS * 3 / 4]
  uint8_t *f8out,        // physical shape: [R, KO]
  // cute::Tensor<cute::ViewEngine<sf_t*>, normal::LayoutSFA> f4scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, sensitive::LayoutSFA> f6scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, outlier::LayoutSFA> f8scale,
  auto f4scale,
  auto f6scale,
  auto f8scale,
  int f4scaleldm,
  int f6scaleldm,
  int f8scaleldm,
  int KN, int KS, int KO
){
  // static_assert(GROUP_SIZE == 32 && HIDDEN_DIM == 4096, "Current only support 32x4096.");
  // static_assert(bdx == 128, "Current 128 threads per block.");
  // static_assert(bdx == HIDDEN_DIM / GROUP_SIZE, "Current only support 4096/32.");
  // [Stage 1: work partition]
  // GROUP_SIZE = 32, HIDDEN_DIM = D, bdx = D / 32.
  // thread block 하나가 input[row_id, :] 한 행 [D]를 담당한다.
  // MXFP scale 하나가 연속된 32개 값에 대응하므로 thread 하나가 32개를 담당한다.
  // 따라서 thread block 전체가 맡는 값은 (D/32 threads) * (32 values/thread) = D개이다.
  constexpr int elements_per_thread = GROUP_SIZE;

  // CUDA Thread Block <-> PTX CTA 대응되는 개념
  cg::thread_block cta = cg::this_thread_block();

  // input_smem BF16, logical shape: [D]. 현재 행 input[row_id, :] 전체를 저장한다.
  __shared__ uint8_t smem[HIDDEN_DIM * sizeof(bf16_t)];
  bf16_t *input_smem = reinterpret_cast<bf16_t*>(smem);

  // input_frag logical shape: [32]. Group quantization의 group
  bf16_t input_frag[elements_per_thread];

  // blockIdx.x가 input [R, D]의 row를 선택한다.
  // input은 input[row_id, 0], output은 각 packed row의 시작 주소로 이동한다.
  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;                              // + row_id * D BF16
  f4out = f4out + row_id * (GROUP_SIZE * GROUP_NUM(KN)) / 2;       // + row_id * KN/2 bytes
  f6out = f6out + row_id * (GROUP_SIZE * GROUP_NUM(KS)) / 4 * 3;   // + row_id * KS*3/4 bytes
  f8out = f8out + row_id * (GROUP_SIZE * GROUP_NUM(KO));           // + row_id * KO bytes

  // [Stage 2: global input[row_id, :] [D] -> shared input_smem [D]]
  int tx = threadIdx.x;
  int tid = tx;
  // reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>
  // ㄴ bdx = hidden_dim / 32: blockDim.x
  constexpr int bytes_per_iter = bdx * 16; // D/2
  constexpr int iters = HIDDEN_DIM * sizeof(bf16_t) / bytes_per_iter; // D*2/ (D/2) = 4
  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterN;
  cutlass::NumericConverter<fp6_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterS;
  cutlass::NumericConverter<fp8_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterO;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
  cutlass::NumericConverter<bf16_t, int, cutlass::FloatRoundStyle::round_to_nearest> converterBF;
  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterScale;
  cutlass::NumericConverter<int, fp4_t, cutlass::FloatRoundStyle::round_to_nearest> converter4i;

  #pragma unroll
  for(int i = 0;i < iters;++i){ // iters = 4
    // 한 row의 size = 2D bytes
    // float4 = float 4개 -> 16 bytes
    // iteration마다 thread 하나가 BF16 8개 = 16 bytes를 sram으로 보냄
    // tid 하나만 놓고 보면, [16 * tid, D/2 + tid * 16, D + tid * 16, 3D/2 + tid * 16] 위치에서 BF16 8개를 복사
    // 그러면, offset ∈ [0, D/2)에서 0...bdx-1번째 thread가 0번째 iter에서 16btyes씩 SRAM으로 복사하고, 
    // 마찬가지로 [D/2, D)에서 0...bdx-1번째 thread가 1번째 iter에서 16bytes씩 SRAM으로 복사하므로 coalesced access 달성!
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  // 모든 input_smem[0:D]가 준비된 뒤 reorder를 시작한다.
  cta.sync();

  // [Stage 3: reorder]
  // channel index를 Activation magnitude에 따라 sort하고, 
  // 이의 original index를 복원하기 위해 reorder_index 배열을 사용함
  /*
  1) Original 
  index:    0    1    2    3    4          126   127
          ┌────┬────┬────┬────┬────┬─ ··· ┬────┬────┐
  value:  │ 0  │  1 │  2 │  3 │  4 │      │126 │127 │
          └────┴────┴────┴────┴────┴─ ··· ┴────┴────┘
  2) Reordered (magnitude ascending)
  index:     0    1    2    3    4          126  127
          ┌────┬────┬────┬────┬────┬─ ··· ┬────┬────┐
  value:  │  7 │  2 │100 │ 14 │  0 │      │  8 │ 51 │
          └────┴────┴────┴────┴────┴─ ··· ┴────┴────┘
  */
  // 그럼 각 tid가 [tid * 32, tid * 32 + 32) 범위에서 32개의 reordered channel을 reg input_frag에 저장
  #pragma unroll 4
  for(int i = 0;i < elements_per_thread;++i){ // GROUP_SIZE
    int offset = tid * GROUP_SIZE + i;
    input_frag[i] = input_smem[reorder_index[offset]];
  }
  // [Stage 4: 32-value block absmax]
  // input_frag [32]에서 scale 계산에 사용할 max(abs(x)) 하나를 구한다.
  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);
  float *input_frag_float = reinterpret_cast<float *>(input_frag);
  constexpr int float4_per_thread = elements_per_thread * sizeof(bf16_t) / sizeof(float4);
  float maxv = 0,  scale = 1.0, r_scale = 1.0;

  // TODO: 아래 unroll 최적화기법 이해하기
  #pragma unroll
  for(int i = 0; i < float4_per_thread;++i){
    maxv = local_abs_max<bf16_t, float4>(input_frag_float4 + i, maxv);
  }
  cta.sync();
  // [Stage 5: precision 선택 + scale 저장]
  // reordered D축: [0, KN)=FP4, [KN, KN+KS)=FP6, [KN+KS, KN+KS+KO)=FP8
  // thread 하나가 32개를 맡으므로 thread 구간도 KN/32, KS/32, KO/32로 나뉜다.
  // ㄴ 각각 logical shape은 [R, KN/32], [R, KS/32], [R, KO/32]가 되겠지
  // 각 block의 power-of-two scale을 CUTLASS SFA/SFB physical layout에 직접 저장한다.
  // ㄴ SFA = Scale factor of A. 걍 A @ B^t GEMM의 operand A, B
  // int replicated_row_id = scale_index(row_id);
  float lower_bound, upper_bound;
  if (tid >= bdx - GROUP_NUM(KO)) {
    // tid = threadIdx.x = 현재 block 안에서 실행중인 thread index ∈ [0, bdx)
    // bdx = blockDim.x = 현재 block 안에서 실행중인 thread 수
    // fp8 quantize
    lower_bound = -FP8_MAX;
    upper_bound = FP8_MAX;
    if (maxv == 0) scale = 0.5; // maxv = 0이면 scale = 0이되고, 그럼 dequant에서 reciprocal이 Div by zero이므로 0.5
    // 애당초 maxv = 0이면 해당 row의 block안의 모든 element값이 0이라 상관없음
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP8_MAX)))));
    int idx = (tid + GROUP_NUM(KO) - bdx);
    // TODO: CUTLASS SF Layout 이해하기
    // https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f8scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)) {
    // fp6 quant
    lower_bound = -FP6_MAX;
    upper_bound = FP6_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP6_MAX)))));
    int idx = (tid + GROUP_NUM(KO + KS) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f6scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else {
    // fp4 quant
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
    auto logical_coord2 = make_coord(0, 0);
    f4scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }

  // [Stage 6: scale 적용 + quantization + packing]
  // input_frag [32]를 scale로 나눈 뒤 thread가 속한 구간의 format으로 변환한다.
  // FP4: 2 values/byte, FP6: 4 values/3 bytes, FP8: 1 value/byte.
  // division 대신 reciprocal multiplication을 사용한다.
  r_scale = 1.0 / scale;

  // Quantize each thread's value
  // int lower_bound = (ty == bdy - 1) ? -128 : -8;
  // int upper_bound = (ty == bdy - 1) ? 127 : 7;
  // Each iteration quantize two things, convenient for packing int4
  fp8_t* input_frag_fp8 = reinterpret_cast<fp8_t*>(input_frag);
  uint8_t* input_frag_fp6 = reinterpret_cast<uint8_t*>(input_frag);
  PackFp4* input_frag_fp4 = reinterpret_cast<PackFp4*>(input_frag);
  for(int i = 0; i < elements_per_thread; i += 4){
    bf16_t result_0, result_1, result_2, result_3;
    result_0 = converterScale(clamp(((float)input_frag[i + 0] * r_scale), lower_bound, upper_bound));
    result_1 = converterScale(clamp(((float)input_frag[i + 1] * r_scale), lower_bound, upper_bound));
    result_2 = converterScale(clamp(((float)input_frag[i + 2] * r_scale), lower_bound, upper_bound));
    result_3 = converterScale(clamp(((float)input_frag[i + 3] * r_scale), lower_bound, upper_bound));
    if(tid >= bdx - GROUP_NUM(KO)){
      input_frag_fp8[i + 0] = converterO(result_0);
      input_frag_fp8[i + 1] = converterO(result_1);
      input_frag_fp8[i + 2] = converterO(result_2);
      input_frag_fp8[i + 3] = converterO(result_3);
    }
    else if(tid >= bdx - GROUP_NUM(KO + KS)) {
      pack_4_fp6_to_3_bytes(
        converterS(result_0).storage, // Corrected
        converterS(result_1).storage, // Corrected
        converterS(result_2).storage, // Corrected
        converterS(result_3).storage, // Corrected
        (input_frag_fp6 + (i / 4) * 3)
      );
    }
    else {
      input_frag_fp4[i / 2].low = converterN(result_0).storage;
      input_frag_fp4[i / 2].high = converterN(result_1).storage;
      input_frag_fp4[i / 2 + 1].low = converterN(result_2).storage;
      input_frag_fp4[i / 2 + 1].high = converterN(result_3).storage;
    }
  }
  // [Stage 7: packed global store]
  // thread가 만든 32개 packed 결과를 현재 row의 Global memory (f4out/f6out/f8out)에 연속 저장한다.
  // Unpack이 안보이는 이유는 CUTLASS/Tensor Core 경로 내부에서 packed FP4/FP6를 직접 consume하기 때문
  if(tid >= bdx - GROUP_NUM(KO)){
    // Store fp8_t quantized result
    float4* f8out_float4 = reinterpret_cast<float4*>(f8out);
    f8out_float4[(tid + GROUP_NUM(KO) - bdx) * 2 + 0] = input_frag_float4[0];
    f8out_float4[(tid + GROUP_NUM(KO) - bdx) * 2 + 1] = input_frag_float4[1];
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)){ // FP6 data processing path
    int idx = (tid + GROUP_NUM(KO + KS) - bdx);
    int64_t* f6out_ll = reinterpret_cast<int64_t*>(f6out);
    int64_t* input_frag_ll = reinterpret_cast<int64_t*>(input_frag);
    f6out_ll[idx * 3 + 0] = input_frag_ll[0];
    f6out_ll[idx * 3 + 1] = input_frag_ll[1];
    f6out_ll[idx * 3 + 2] = input_frag_ll[2];
  }
  else {
    // Store fp4_t quantized result
    float4* f4out_float4 = reinterpret_cast<float4*>(f4out);
    f4out_float4[tid] = input_frag_float4[0];
  }
}

// f62c4c2 QLinearLayer가 실제 사용하는 weight preprocessing kernel.
// reorder_quantize_mixed_kernel() 과는 다르게 모두 FP4로 Quantize
template <int bdx, int GROUP_SIZE, int HIDDEN_DIM>
__global__ void reorder_quantize_mxfp4_kernel(
  bf16_t *input,          // W [H, D]
  int16_t *reorder_index,// [D]
  uint8_t *f4out,        // [H, KN/2]
  uint8_t *f6out,        // [H, KS/2], 이름과 달리 저장 format은 FP4
  uint8_t *f8out,        // [H, KO/2], 이름과 달리 저장 format은 FP4
  // cute::Tensor<cute::ViewEngine<sf_t*>, normal::LayoutSFB> f4scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, sensitive::LayoutSFB> f6scale,
  // cute::Tensor<cute::ViewEngine<sf_t*>, outlier::LayoutSFB> f8scale,
  auto f4scale,
  auto f6scale,
  auto f8scale,
  int f4scaleldm,
  int f6scaleldm,
  int f8scaleldm, 
  int KN, int KS, int KO
){
  // static_assert(GROUP_SIZE == 32 && HIDDEN_DIM == 4096, "Current only support 32x4096.");
  // static_assert(bdx == 128, "Current 128 threads per block.");
  // static_assert(bdx == HIDDEN_DIM / GROUP_SIZE, "Current only support 4096/32.");
  // [Stage 1] H개의 thread block 중 block row_id가 W[row_id, :] [D] 하나를 담당한다.
  // D/32개 thread 각각이 reordered weight 32개와 scale 하나를 담당한다.
  constexpr int elements_per_thread = GROUP_SIZE;

  cg::thread_block cta = cg::this_thread_block();

  // input_smem [D]: 현재 output channel의 weight row W[row_id, :].
  __shared__ uint8_t smem[HIDDEN_DIM * sizeof(bf16_t)];
  bf16_t *input_smem = reinterpret_cast<bf16_t*>(smem);

  // input_frag [32]: 현재 thread가 reorder 후 FP4로 양자화할 weight 32개.
  bf16_t input_frag[elements_per_thread];

  // input과 세 packed output pointer를 row_id번째 row의 시작점으로 이동한다.
  int row_id = blockIdx.x;
  input = input + row_id * HIDDEN_DIM;
  f4out = f4out + row_id * (GROUP_SIZE * GROUP_NUM(KN)) / 2;
  f6out = f6out + row_id * (GROUP_SIZE * GROUP_NUM(KS)) / 2;
  f8out = f8out + row_id * (GROUP_SIZE * GROUP_NUM(KO)) / 2;

  // [Stage 2] W[row_id, :] [D]를 global memory에서 shared memory로 연속 load한다.
  int tx = threadIdx.x;
  int tid = tx;
  constexpr int bytes_per_iter = bdx * 16;
  constexpr int iters = HIDDEN_DIM * sizeof(bf16_t) / bytes_per_iter;
  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterN;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterSF;
  cutlass::NumericConverter<bf16_t, int, cutlass::FloatRoundStyle::round_to_nearest> converterBF;
  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> converterScale;

  #pragma unroll
  for(int i = 0;i < iters;++i){
    // Each thread loads 16 bytes
    int offset = i * bytes_per_iter + tid * 16;
    *(float4 *)(reinterpret_cast<uint8_t *>(input_smem) + offset) = *(float4 *)(reinterpret_cast<uint8_t *>(input) + offset);
  }
  cta.sync();
  // [Stage 3] shared memory에서 reorder_index를 적용해 thread별 input_frag [32]를 만든다.
  #pragma unroll 4
  for(int i = 0;i < elements_per_thread;++i){
    int offset = tid * GROUP_SIZE + i;
    input_frag[i] = input_smem[reorder_index[offset]];
  }
  // [Stage 4] input_frag [32]의 max(abs(x))를 계산한다.
  float4 *input_frag_float4 = reinterpret_cast<float4 *>(input_frag);
  float *input_frag_float = reinterpret_cast<float *>(input_frag);
  constexpr int float4_per_thread = elements_per_thread * sizeof(bf16_t) / sizeof(float4);
  float maxv = 0, scale = 1.0, r_scale = 1.0;

  #pragma unroll
  for(int i = 0; i < float4_per_thread;++i){
    maxv = local_abs_max<bf16_t, float4>(input_frag_float4 + i, maxv);
  }
  cta.sync();
  // [Stage 5] 모든 구간에 MXFP4 scale을 계산한다.
  // KN/KS/KO는 activation A4/A6/A8과 각각 GEMM하기 위해 buffer와 SFB layout만 분리한다.
  // int replicated_row_id = scale_index(row_id);
  float lower_bound, upper_bound;
  if (tid >= bdx - GROUP_NUM(KO)) {
    // fp4 quantize
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    int idx = (tid + GROUP_NUM(KO) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f8scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)) {
    // fp4 quant
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    int idx = (tid + GROUP_NUM(KO + KS) - bdx);
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, idx % 4), idx / 4);
    auto logical_coord2 = make_coord(0, 0);
    f6scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }
  else {
    // fp4 quant
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    if (maxv == 0) scale = 0.5;
    else scale = converterScale(ldexpf(1.0f, static_cast<int>(ceil(log2(maxv / FP4_MAX)))));
    auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    auto logical_coord1 = make_coord(make_coord(0, tid % 4), tid / 4);
    auto logical_coord2 = make_coord(0, 0);
    f4scale(make_coord(logical_coord0, logical_coord1, logical_coord2)) = converterSF(scale);
  }

  // [Stage 6] 세 구간 모두 FP4로 양자화하고 2 values/byte로 packing한다.
  r_scale = 1.0 / scale;

  // Quantize each thread's value
  // int lower_bound = (ty == bdy - 1) ? -128 : -8;
  // int upper_bound = (ty == bdy - 1) ? 127 : 7;
  // Each iteration quantize two things, convenient for packing int4
  // fp4_t* input_frag_fp8 = reinterpret_cast<fp4_t*>(input_frag);
  // fp4_t* input_frag_fp6 = reinterpret_cast<fp4_t*>(input_frag);
  // fp4_t* input_frag_fp4 = reinterpret_cast<fp4_t*>(input_frag);
  PackFp4* input_frag_fp4 = reinterpret_cast<PackFp4*>(input_frag);
  PackFp4* input_frag_fp6 = reinterpret_cast<PackFp4*>(input_frag);
  PackFp4* input_frag_fp8 = reinterpret_cast<PackFp4*>(input_frag);
  
  for(int i = 0; i < elements_per_thread; i += 2){
    bf16_t result_0, result_1;
    result_0 = converterScale(clamp(input_frag[i] * r_scale, lower_bound, upper_bound));
    result_1 = converterScale(clamp(input_frag[i + 1] * r_scale, lower_bound, upper_bound));
    if(tid >= bdx - GROUP_NUM(KO)){
      input_frag_fp8[i / 2].low = converterN(result_0).storage;
      input_frag_fp8[i / 2].high = converterN(result_1).storage;
    }
    else if(tid >= bdx - GROUP_NUM(KO + KS)) {
      input_frag_fp6[i / 2].low = converterN(result_0).storage;
      input_frag_fp6[i / 2].high = converterN(result_1).storage;
    }
    else {
      input_frag_fp4[i / 2].low = converterN(result_0).storage;
      input_frag_fp4[i / 2].high = converterN(result_1).storage;
    }
  }
  // [Stage 7] packed FP4를 [H, KN/2], [H, KS/2], [H, KO/2] buffer에 저장한다.
  if(tid >= bdx - GROUP_NUM(KO)){
    // Store fp4_t quantized result
    float4* f8out_float4 = reinterpret_cast<float4*>(f8out);
    f8out_float4[(tid + GROUP_NUM(KO) - bdx)] = input_frag_float4[0];
  }
  else if(tid >= bdx - GROUP_NUM(KO + KS)){
    // Store fp4_t quantized result
    float4* f6out_float4 = reinterpret_cast<float4*>(f6out);
    f6out_float4[(tid + GROUP_NUM(KO + KS) - bdx)] = input_frag_float4[0];
  }
  else {
    // Store fp4_t quantized result
    float4* f4out_float4 = reinterpret_cast<float4*>(f4out);
    f4out_float4[tid] = input_frag_float4[0];
  }
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_x(
  bf16_t *hidden_states,
  int seq_len,
  // int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
){
  // static_assert(group_size == 32 && hidden_dim == 4096, "Current only support 32x4096.");
  // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
  // Activation X [S, D]: seq_len=S, hidden_dim=D.
  // grid=[S]이므로 thread block 하나가 X[s, :]를 처리하고, block=[D/32]이므로 thread 하나가 32개를 처리한다.
  // Activation은 GEMM의 A operand이므로 scale을 LayoutSFA로 만든다.
  dim3 grids(seq_len);       // S thread blocks
  dim3 blocks(hidden_dim / 32); // D/32 threads per block
  Tensor sfan_tensor = cute::make_tensor(normal_scale, filter_zeros(normal::get_layoutSFA(seq_len, KN)));
  Tensor sfas_tensor = cute::make_tensor(sensitive_scale, filter_zeros(sensitive::get_layoutSFA(seq_len, KS)));
  Tensor sfao_tensor = cute::make_tensor(outlier_scale, filter_zeros(outlier::get_layoutSFA(seq_len, KO)));
  reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (int16_t *)reorder_index,
    (uint8_t *)o_normal,
    (uint8_t *)o_sensitive,
    (uint8_t *)o_outlier,
    sfan_tensor,
    sfas_tensor,
    sfao_tensor,
    SCALE_SIZE(KN),
    SCALE_SIZE(KS),
    SCALE_SIZE(KO),
    KN, KS, KO
  );
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_w(
  bf16_t *hidden_states,
  // int seq_len,
  int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
){
  // static_assert(group_size == 32 && hidden_dim == 4096, "Current only support 32x4096.");
  // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
  // Weight W [H, D]: out_features=H, hidden_dim=D.
  // grid=[H], block=[D/32]. Weight는 GEMM의 B operand이므로 scale은 LayoutSFB를 사용한다.
  // 이 matching FP4/FP6/FP8 weight 경로는 f62c4c2 QLinearLayer가 현재 호출하지 않는다.
  dim3 grids(out_features);     // H thread blocks
  dim3 blocks(hidden_dim / 32); // D/32 threads per block
  Tensor sfbn_tensor = cute::make_tensor(normal_scale, filter_zeros(normal::get_layoutSFB(out_features, KN)));
  Tensor sfbs_tensor = cute::make_tensor(sensitive_scale, filter_zeros(sensitive::get_layoutSFB(out_features, KS)));
  Tensor sfbo_tensor = cute::make_tensor(outlier_scale, filter_zeros(outlier::get_layoutSFB(out_features, KO)));
  reorder_quantize_mixed_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (int16_t *)reorder_index,
    (uint8_t *)o_normal,
    (uint8_t *)o_sensitive,
    (uint8_t *)o_outlier,
    sfbn_tensor,
    sfbs_tensor,
    sfbo_tensor,
    SCALE_SIZE(KN),
    SCALE_SIZE(KS),
    SCALE_SIZE(KO),
    KN, KS, KO
  );
}

template<int group_size, int hidden_dim>
void run_reorder_quantize_w4(
  bf16_t *hidden_states,
  // int seq_len,
  int out_features,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
){
  // static_assert(group_size == 32 && hidden_dim == 4096, "Current only support 32x4096.");
  // static_assert(KN % 128 == 0 && KS % 128 == 0 && KO % 128 == 0, "TMA requires 32bytes alignment.");
  // Weight W [H, D]를 세 구간 모두 MXFP4로 만드는 f62c4c2 실제 경로.
  // grid=[H], block=[D/32], scale layout은 B operand용 LayoutSFB이다.
  dim3 grids(out_features);     // H thread blocks
  dim3 blocks(hidden_dim / 32); // D/32 threads per block
  Tensor sfbn_tensor = cute::make_tensor(normal_scale, filter_zeros(normal::get_layoutSFB(out_features, KN)));
  Tensor sfbs_tensor = cute::make_tensor(sensitive_scale, filter_zeros(sensitive::get_layoutSFB(out_features, KS)));
  Tensor sfbo_tensor = cute::make_tensor(outlier_scale, filter_zeros(outlier::get_layoutSFB(out_features, KO)));
  reorder_quantize_mxfp4_kernel<hidden_dim / 32, group_size, hidden_dim><<<grids, blocks>>>(
    (bf16_t *)hidden_states,
    (int16_t *)reorder_index,
    (uint8_t *)o_normal,
    (uint8_t *)o_sensitive,
    (uint8_t *)o_outlier,
    sfbn_tensor,
    sfbs_tensor,
    sfbo_tensor,
    SCALE_SIZE(KN),
    SCALE_SIZE(KS),
    SCALE_SIZE(KO),
    KN, KS, KO
  );
}

template void run_reorder_quantize_x<32, 4096>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 4096>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 4096>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 3584>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 3584>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 3584>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 3072>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 3072>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 3072>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 5120>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 5120>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 5120>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 14336>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 14336>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 14336>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);


template void run_reorder_quantize_x<32, 18944>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 18944>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 18944>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 11008>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 11008>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 11008>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 12288>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 12288>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 12288>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 13824>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 13824>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 13824>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_x<32, 8192>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w<32, 8192>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

template void run_reorder_quantize_w4<32, 8192>(
  bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
  sf_t*, sf_t*, sf_t*, int, int, int
);

// template void run_reorder_quantize_x<32, 27648>(
//   bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
//   sf_t*, sf_t*, sf_t*, int, int, int
// );

// template void run_reorder_quantize_w<32, 27648>(
//   bf16_t*, int, int16_t*, uint8_t*, uint8_t*, uint8_t*,
//   sf_t*, sf_t*, sf_t*, int, int, int
// );
