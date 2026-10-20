# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    "-O3",
]

# NDEBUG is opt-OUT, because turning it off is a debugging tool with real value:
# CUTLASS guards its parameter checks with it, so a misconfigured pipeline fails
# SILENTLY in the shipping build.  Concretely, sm90_pipeline.hpp:355-370 sets
# `is_signalling_thread_ = 0` and then `asm("brkpt")` when `num_consumers` matches
# neither 32 nor a multiple of 128 -- with NDEBUG that brkpt is gone and the only
# symptom is `unspecified launch failure`, which cost four rounds of guessing.
#
# ⚠️ NEVER take PERFORMANCE numbers from an assert-enabled build: the asserts cost
# instructions and perturb register allocation.  Use it to localize, then turn it
# back on.  Usage:  MEGA_MLA_ASSERTS=1 python setup.py build_ext --inplace
if os.environ.get("MEGA_MLA_ASSERTS", "0") in ("0", "", "false", "False"):
    gencode_flags.append("-DNDEBUG")
else:
    print("[setup.py] NDEBUG OFF -- CUTLASS asserts enabled (do NOT trust timings)")

# Opt-in debug defines, e.g. MEGA_MLA_NVCC_DEFINES="-DMEGA_MLA_WATCHDOG=1".
# Kept as an env passthrough rather than a permanent flag so the shipping build is
# byte-identical when it is unset.
_extra = os.environ.get("MEGA_MLA_NVCC_DEFINES", "").split()
if _extra:
    print(f"[setup.py] extra nvcc defines: {' '.join(_extra)}")
    gencode_flags += _extra

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
                "csrc/kernel/mega_mla/mla_plan.cu",
                "csrc/kernel/mega_mla/mega_mla.cu",
                "csrc/common/cpp_extensions.cc",
            ],
            # ⚠️ The debug defines must reach the C++ TU too, not just nvcc: the
            # pybind registration in cpp_extensions.cc is guarded by the same macros,
            # and a define that reaches only nvcc yields a binding that is declared in
            # the .cu and never registered -- the symptom is "the extension does not
            # expose <name>" on a build that compiled cleanly.
            extra_compile_args={"nvcc": gencode_flags, "cxx": _extra},
            libraries=["cublasLt", "cuda", "nvidia-ml"],
        ),
    ],
    cmdclass={"build_ext": InplaceBuildExtension},
)
