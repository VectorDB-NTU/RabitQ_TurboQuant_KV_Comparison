"""
Build rabitq Python extension.

Usage:
    cd llm_rabitq
    pip install -e .
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_sources = [
    "binding.cpp",
    "inc/quantizer_standalone.cu",
    "inc/rotator_gpu.cu",
    "inc/fht_kac_rotator_gpu.cu",
    "inc/quantizer_gpu_fast.cu",
]

include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "inc"),
]

for d in ["/usr/include/eigen3", "/usr/local/include/eigen3"]:
    if os.path.isdir(d):
        include_dirs.append(d)
        break

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
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
