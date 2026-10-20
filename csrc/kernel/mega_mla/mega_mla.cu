// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mega_mla_kernel.cuh"

static_assert(
    MEGA_MLA_ARCH_GATE_CHECKED,
    "mla_intrinsics.cuh must define the KERUTILS_ENABLE_SM100A gate before "
    "this file is compiled: without it the whole kernel body is preprocessed "
    "away and the build SUCCEEDS with an empty kernel.");

namespace halops {

namespace {

using MlaWorkspace = MegaMlaWorkspace<mega_mla::TraitsN128>;

template <class KT>
void mega_mla_launch(const MegaMlaArguments& args) {
    const MegaMlaProblem<KT> prob = MegaMlaProblem<KT>::from(args);
    const auto mp =
        mega_mla::CollectiveMainloop<KT>::to_underlying_arguments(args, prob);
    const auto ep =
        mega_mla::CollectiveEpilogue<KT>::to_underlying_arguments(args, prob);
    const auto sched =
        mega_mla::MlaScheduler<KT>::to_underlying_arguments(prob, mp, ep);

    const size_t smem_size = sizeof(SharedMemoryPlan<KT>);
    auto* kernel =
        mega_mla_kernel<KT, decltype(mp), decltype(ep), decltype(sched)>;
    if (smem_size > 48 * 1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(prob.grid_x, 1, 1), KT::NumThreads, smem_size,
             c10::cuda::getCurrentCUDAStream()>>>(mp, ep, sched);
    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::vector<int64_t> MegaMlaWorkspaceBytes(int64_t num_tokens,
                                           torch::Tensor wo_a_weight,
                                           torch::Tensor wo_b_weight,
                                           torch::Tensor wq_a_weight,
                                           torch::Tensor wkv_weight) {
    using D = mega_mla::ModelDims;
    const int T = (int)num_tokens;
    TORCH_CHECK(T > 0, "mega_mla_workspace_bytes: num_tokens must be positive");
    auto have = [](torch::Tensor const& w) { return w.defined() && w.numel(); };
    using P = MegaMlaProblem<mega_mla::TraitsN128>;
    const int o_groups = (int)((int64_t)D::O_TOTAL / P::w_cols(wo_a_weight));
    const int o_lora_rank = P::w_rows(wo_a_weight) / o_groups;
    const int hidden = P::w_rows(wo_b_weight);
    const int qlora =
        have(wq_a_weight) ? P::w_rows(wq_a_weight) : (int)D::QLORA_RANK;
    const int kvlora = have(wkv_weight) ? P::w_rows(wkv_weight) : (int)D::D_V;

    const MlaWorkspace::Dimensions wsd =
        MlaWorkspace::dims(hidden, qlora, kvlora, o_groups * o_lora_rank);
    const MlaWorkspace::Geom wsg = MlaWorkspace::geom(
        T, mega_mla::grid_x_for(T), mega_mla::n_groups_for(T),
        olora_rounds_from_wo_a(wo_a_weight));
    return {wsd.bytes(wsg, MlaWorkspace::token_pad(T)), wsg.token_off};
}

void MegaMla(torch::Tensor q_or_x, torch::Tensor kv_cache, int64_t page_size,
             torch::Tensor kv_runs, int64_t topk, int64_t kv_run_step,
             torch::Tensor topk_lengths, torch::Tensor positions,
             torch::Tensor freqs_real, torch::Tensor wo_a_weight,
             torch::Tensor wo_a_scale_inv, torch::Tensor wo_b_weight,
             torch::Tensor out, double sm_scale, torch::Tensor wq_a_weight,
             torch::Tensor q_norm_weight, torch::Tensor wq_b_weight,
             double q_eps, bool use_phase1, torch::Tensor wkv_weight,
             torch::Tensor kv_norm_weight, double kv_eps, bool use_kv_write,
             torch::Tensor swa_loc, torch::Tensor attn_sink,
             torch::Tensor wq_a_sf_g32, torch::Tensor wq_b_sf_g32,
             torch::Tensor wkv_sf_g32, torch::Tensor wo_b_sf_g32,
             torch::Tensor wo_a_sf_g32, torch::Tensor workspace,
             torch::Tensor wkv_gate_weight, torch::Tensor kv_score_ring,
             torch::Tensor kv_gate_ring_row, torch::Tensor c128_ape,
             torch::Tensor c128_norm_weight, torch::Tensor c128_freqs_real,
             torch::Tensor c128_out_loc, torch::Tensor c128_row_out,
             torch::Tensor c128_pool, int64_t c128_stride_kv_block,
             int64_t c128_page_block, torch::Tensor c128_attn_slot0,
             torch::Tensor c128_attn_len, int64_t draft_tokens) {
    const MegaMlaArguments args{q_or_x,
                                kv_cache,
                                page_size,
                                kv_runs,
                                topk,
                                kv_run_step,
                                topk_lengths,
                                positions,
                                freqs_real,
                                wo_a_weight,
                                wo_a_scale_inv,
                                wo_b_weight,
                                out,
                                sm_scale,
                                wq_a_weight,
                                q_norm_weight,
                                wq_b_weight,
                                q_eps,
                                use_phase1,
                                wkv_weight,
                                kv_norm_weight,
                                kv_eps,
                                use_kv_write,
                                swa_loc,
                                attn_sink,
                                wq_a_sf_g32,
                                wq_b_sf_g32,
                                wkv_sf_g32,
                                wo_b_sf_g32,
                                wo_a_sf_g32,
                                workspace,
                                wkv_gate_weight,
                                kv_score_ring,
                                kv_gate_ring_row,
                                c128_ape,
                                c128_norm_weight,
                                c128_freqs_real,
                                c128_out_loc,
                                c128_row_out,
                                c128_pool,
                                c128_stride_kv_block,
                                c128_page_block,
                                c128_attn_slot0,
                                c128_attn_len,
                                draft_tokens};
    switch (mega_mla::select_tile_t((int)args.kv_runs.size(0))) {
        case 16:
            mega_mla_launch<mega_mla::TraitsN16>(args);
            break;
        case 32:
            mega_mla_launch<mega_mla::TraitsN32>(args);
            break;
        case 64:
            mega_mla_launch<mega_mla::TraitsN64>(args);
            break;
        default:
            mega_mla_launch<mega_mla::TraitsN128>(args);
            break;
    }
}

#ifdef MEGA_MLA_TRACE
torch::Tensor MegaMlaReadTrace() {
    using Tr = mega_mla::KernelTrait::Trace;
    constexpr int N = Tr::CtaSlots * Tr::WarpsPerCta * Tr::WordsPerWarp;
    static cudaStream_t probe_stream = nullptr;
    if (probe_stream == nullptr)
        C10_CUDA_CHECK(
            cudaStreamCreateWithFlags(&probe_stream, cudaStreamNonBlocking));
    auto out = torch::empty(
        {N},
        torch::dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    C10_CUDA_CHECK(cudaMemcpyFromSymbolAsync(
        out.data_ptr(), mega_mla::mega_mla_trace, sizeof(int) * N, 0,
        cudaMemcpyDeviceToHost, probe_stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(probe_stream));
    return out;
}

void MegaMlaClearTrace() {
    using Tr = mega_mla::KernelTrait::Trace;
    constexpr int N = Tr::CtaSlots * Tr::WarpsPerCta * Tr::WordsPerWarp;
    void* dptr = nullptr;
    C10_CUDA_CHECK(cudaGetSymbolAddress(&dptr, mega_mla::mega_mla_trace));
    C10_CUDA_CHECK(cudaMemset(dptr, 0, sizeof(int) * N));
}
#endif

}  // namespace halops
