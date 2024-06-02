import os
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys, subprocess

__version__ = "0.0.1"

CURRENT_DIR = os.getcwd()
SOURCE_PATH = os.path.join(CURRENT_DIR, "python_bindings.cpp")


def simd_extension_supported(extension: str) -> bool:
    """
    TODO: Make this check more robust across compilers and supported platforms.
    """
    if sys.platform in ["linux", "linux2", "darwin"]:
        try:
            output = subprocess.check_output("cat /proc/cpuinfo", shell=True)
            decoded_output = output.decode()
            return extension in decoded_output
        except Exception:
            return False

    else:
        return False


INCLUDE_DIRS = [
    os.path.join(CURRENT_DIR, ".."),
    os.path.join(CURRENT_DIR, "..", "external", "cereal", "include"),
]
EXTRA_LINK_ARGS = []

if sys.platform == "darwin":
    omp_flag = "-Xclang -fopenmp"
    INCLUDE_DIRS.extend(["/opt/homebrew/opt/libomp/include"])
    EXTRA_LINK_ARGS.extend(["-lomp", "-L/opt/homebrew/opt/libomp/lib"])
elif sys.platform == "linux":
    omp_flag = "-fopenmp"
    EXTRA_LINK_ARGS.extend(["-fopenmp"])


EXTRA_COMPILE_ARGS = [
    omp_flag,  # Enable OpenMP
    "-Ofast",  # Use the fastest optimization
    "-fpic",  # Position-independent code
    "-w",  # Suppress all warnings (note: this overrides -Wall)
    "-ffast-math",  # Enable fast math optimizations
    "-funroll-loops",  # Unroll loops
]

# We don't include SIMD flags if the NO_SIMD_VECTORIZATION variable is set to 1
no_simd_vectorization = int(os.environ.get("NO_SIMD_VECTORIZATION", "0"))

if not no_simd_vectorization:
    SIMD_EXTENSIONS = ["sse", "sse3", "avx", "avx512f", "avx512bw"]
    found_single_extension = False
    for extension in SIMD_EXTENSIONS:
        if simd_extension_supported(extension=extension):
            EXTRA_COMPILE_ARGS.append(f"-m{extension}")
            found_single_extension = True

    if not found_single_extension:
        # We've found that ftree-vectorize (auto-vectorization) is good in general
        # but it does slow down SIMD extensions considerably. So, we only enable it
        # if we haven't found any SIMD extensions.
        # Reference: https://llvm.org/docs/Vectorizers.html
        EXTRA_COMPILE_ARGS.append("-ftree-vectorize")


ext_modules = [
    Pybind11Extension(
        "flatnav",
        [SOURCE_PATH],
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=17,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,  # Link OpenMP when linking the extension
    )
]


setup(
    name="flatnav",
    version=__version__,
    author="Benjamin Coleman",
    author_email="benjamin.ray.coleman@gmail.com",
    url="https://randorithms.com",
    description="Graph kNN with reordering.",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
