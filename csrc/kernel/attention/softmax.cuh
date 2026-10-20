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

#pragma once
#include "kernel/attention/utils.cuh"

namespace halops {

class Softmax {
  public:
    float row_max = -FLT_MAX, row_sum = 0.0f;
    float const softmax_scale_log2;
    static constexpr float rescale_threshold = 8.0f;
    static constexpr bool e2e = true;
    static constexpr int e2e_freq = 16;
    static constexpr int e2e_res = 4;
    static constexpr int e2e_frg_limit = 1;

    CUTE_DEVICE Softmax(float const softmax_scale_log2)
        : softmax_scale_log2(softmax_scale_log2) {}

    template <bool SplitKV, typename Tensor0>
    CUTE_DEVICE float compute_row_max(Tensor0 const &acc_S_row,
                                      float const row_max_old = -FLT_MAX) {
        float local_max[4] = {
            fmax(acc_S_row[0], acc_S_row[1]), fmax(acc_S_row[2], acc_S_row[3]),
            fmax(acc_S_row[4], acc_S_row[5]), fmax(acc_S_row[6], acc_S_row[7])};
        constexpr int len = cute::size(Tensor0{}) / 8 - 1;
        range_constexpr<len>([&](auto index) {
            constexpr int i = index * 8 + 8;
            local_max[0] =
                fmax_3_input(local_max[0], acc_S_row[i], acc_S_row[i + 1]);
            local_max[1] =
                fmax_3_input(local_max[1], acc_S_row[i + 2], acc_S_row[i + 3]);
            local_max[2] =
                fmax_3_input(local_max[2], acc_S_row[i + 4], acc_S_row[i + 5]);
            local_max[3] =
                fmax_3_input(local_max[3], acc_S_row[i + 6], acc_S_row[i + 7]);
        });
        local_max[0] = fmax_3_input(local_max[0], local_max[1], row_max_old);
        local_max[0] = fmax_3_input(local_max[0], local_max[2], local_max[3]);
        if constexpr (SplitKV) {
            const float neginhbor_max =
                __shfl_xor_sync(0xffffffff, local_max[0], 2);
            local_max[0] = fmax(local_max[0], neginhbor_max);
        }
        return local_max[0];
    }

    template <bool Is_first, bool SplitKV, typename Tensor0>
    CUTE_DEVICE float update_row_max(Tensor0 const &acc_S_row) {
        float acc_scale;
        if constexpr (Is_first) {
            float const row_max_new = compute_row_max<SplitKV>(acc_S_row);
            acc_scale = 0.0;
            row_max = row_max_new;
        } else {
            float row_max_old = row_max;
            float row_max_new =
                compute_row_max<SplitKV>(acc_S_row, row_max_old);
            float const acc_scale_ =
                (row_max_old - row_max_new) * softmax_scale_log2;
            if constexpr (rescale_threshold > 0.0f) {
                if (acc_scale_ >= -rescale_threshold) {
                    row_max_new = row_max_old;
                    acc_scale = 1.0f;
                } else {
                    acc_scale = exp2f(acc_scale_);
                }
            }
            row_max = row_max_new;
        }
        return acc_scale;
    }

    template <bool Is_first, typename Tensor0>
    CUTE_DEVICE void update_row_sum(Tensor0 const &acc_S_row_exp,
                                    float const row_scale) {
        float2 local_sum_0 =
            Is_first ? float2{acc_S_row_exp[0], acc_S_row_exp[1]}
                     : add_packed_f32x2({row_sum * row_scale, 0.0f},
                                        {acc_S_row_exp[0], acc_S_row_exp[1]});
        float2 local_sum_1 = {acc_S_row_exp[2], acc_S_row_exp[3]};
        float2 local_sum_2 = {acc_S_row_exp[4], acc_S_row_exp[5]};
        float2 local_sum_3 = {acc_S_row_exp[6], acc_S_row_exp[7]};
        constexpr int len = cute::size(Tensor0{}) / 8 - 1;
        range_constexpr<len>([&](auto index) {
            constexpr int i = index * 8 + 8;
            local_sum_0 = add_packed_f32x2(
                local_sum_0, {acc_S_row_exp[i + 0], acc_S_row_exp[i + 1]});
            local_sum_1 = add_packed_f32x2(
                local_sum_1, {acc_S_row_exp[i + 2], acc_S_row_exp[i + 3]});
            local_sum_2 = add_packed_f32x2(
                local_sum_2, {acc_S_row_exp[i + 4], acc_S_row_exp[i + 5]});
            local_sum_3 = add_packed_f32x2(
                local_sum_3, {acc_S_row_exp[i + 6], acc_S_row_exp[i + 7]});
        });
        local_sum_0 = add_packed_f32x2(local_sum_0, local_sum_1);
        local_sum_2 = add_packed_f32x2(local_sum_2, local_sum_3);
        local_sum_0 = add_packed_f32x2(local_sum_0, local_sum_2);
        row_sum = local_sum_0.x + local_sum_0.y;
    }

    template <typename Tensor0>
    CUTE_DEVICE void scale_subtract_rowmax(Tensor0 &acc_S_row) {
        float const row_max_scaled = row_max * softmax_scale_log2;
        float2 const row_max_scaled_2 = {-row_max_scaled, -row_max_scaled};
        float2 const softmax_scale_log2_2 = {softmax_scale_log2,
                                             softmax_scale_log2};
#pragma unroll
        for (int i = 0; i < cute::size(acc_S_row); i += 2) {
            fma_packed_f32x2(*reinterpret_cast<float2 *>(&acc_S_row[i]),
                             softmax_scale_log2_2, row_max_scaled_2,
                             *reinterpret_cast<float2 *>(&acc_S_row[i]));
        }
    }

    template <typename Element, typename Tensor0, typename Tensor1>
    CUTE_DEVICE void apply_exp2_convert(Tensor0 &acc_S_row,
                                        Tensor1 &acc_S_row_converted) {
        constexpr int frg_tile = 32;
        constexpr int frg_cnt = cute::size(Tensor0{}) / frg_tile;
        cute::Tensor acc_S_row_frg = cute::logical_divide(
            acc_S_row, cute::make_layout(cute::Int<frg_tile>{}));
        cute::Tensor acc_S_row_converted_frg = cute::logical_divide(
            acc_S_row_converted, cute::make_layout(cute::Int<frg_tile>{}));
        range_constexpr<frg_cnt>([&](auto j) {
            range_constexpr<frg_tile / 2>([&](auto index) {
                constexpr int k = index * 2;
                if constexpr (!e2e) {
                    acc_S_row_frg(k, j) = exp2f(acc_S_row_frg(k, j));
                    acc_S_row_frg(k + 1, j) = exp2f(acc_S_row_frg(k + 1, j));
                } else {
                    if constexpr (k % e2e_freq < e2e_freq - e2e_res ||
                                  j >= frg_cnt - e2e_frg_limit) {
                        acc_S_row_frg(k, j) = exp2f(acc_S_row_frg(k, j));
                        acc_S_row_frg(k + 1, j) =
                            exp2f(acc_S_row_frg(k + 1, j));
                    } else {
                        ex2_emulation_2(acc_S_row_frg(k, j),
                                        acc_S_row_frg(k + 1, j));
                    }
                }
                acc_S_row_converted_frg(k, j) =
                    static_cast<Element>(acc_S_row_frg(k, j));
                acc_S_row_converted_frg(k + 1, j) =
                    static_cast<Element>(acc_S_row_frg(k + 1, j));
            });
        });
    }
};

}  // namespace halops
