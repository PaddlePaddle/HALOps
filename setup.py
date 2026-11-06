"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


class InplaceBuildExtension(BuildExtension):
    def run(self):
        super().run()
        self._copy_extensions_to_source()

    def _copy_extensions_to_source(self):
        import shutil

        for extension in self.extensions:
            relative_path = self.get_ext_filename(extension.name)
            src = os.path.join(self.build_lib, relative_path)
            dst = os.path.join(PROJECT_DIR, relative_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  copied -> {dst}")


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CUTLASS_DIR = os.path.join(PROJECT_DIR, "3rdparty", "cutlass")

if not os.path.exists(os.path.join(CUTLASS_DIR, "include")):
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--depth", "1", "3rdparty/cutlass"],
        cwd=PROJECT_DIR,
    )

gencode_flags = [
    f"-I{PROJECT_DIR}",
    f"-I{PROJECT_DIR}/csrc",
    f"-I{CUTLASS_DIR}/include",
    f"-I{CUTLASS_DIR}/tools/util/include",
    "-std=c++20",
    "-gencode",
    "arch=compute_100f,code=sm_100f",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ptxas-options=-v",
    "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
    "-lineinfo",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-O3",
]

setup(
    ext_modules=[
        CUDAExtension(
            name="halops.halops_kernel",
            sources=[
                "csrc/kernel/attention/encoder_attention.cu",
                "csrc/kernel/attention/cachekv.cu",
                "csrc/kernel/attention/decoder_attention.cu",
                "csrc/kernel/gemm/dense_gemm.cu",
                "csrc/kernel/gemm/group_masked_gemm.cu",
                "csrc/kernel/gemm/group_contiguous_gemm.cu",
                "csrc/kernel/gemm/batch_gemm.cu",
                "csrc/kernel/gemm/group_wise_quant.cu",
                "csrc/kernel/gemm/prefill_permute_to_masked_gemm.cu",
                "csrc/kernel/gemm/fused_swiglu_quant.cu",
                "csrc/kernel/mla/mla_plan.cu",
                "csrc/kernel/mla/mla_attention.cu",
                "csrc/kernel/mla/mla_prequant_x.cu",
                "csrc/kernel/mla/mla_output_proj.cu",
                "csrc/kernel/mla/combine_with_rope_quant.cu",
                "csrc/kernel/indexer/indexer_prepare.cu",
                "csrc/common/cpp_extensions.cc",
            ],
            extra_compile_args={"nvcc": gencode_flags},
            libraries=["cuda"],
        ),
    ],
    cmdclass={"build_ext": InplaceBuildExtension},
)
