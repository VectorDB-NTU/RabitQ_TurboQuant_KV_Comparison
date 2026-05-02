"""
Build rabitq Python extension.

Usage:
    cd llm_rabitq
    pip install -e . --no-build-isolation
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_sources = [
    "binding.cpp",
    "src/quantizer/quantizer_standalone.cu",
    "src/quantizer/rescale_search_gpu.cu",
]

include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "include"),
]

setup(
    name="rabitq",
    ext_modules=[
        CUDAExtension(
            name="rabitq",
            sources=cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--std=c++17", "--expt-relaxed-constexpr", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
