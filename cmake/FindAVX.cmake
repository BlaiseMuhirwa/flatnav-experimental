cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Code adapted from PyTorch:
# https://github.com/pytorch/pytorch/blob/main/cmake/Modules/FindAVX.cmake
include(CheckCXXSourceRuns)
set(AVX_CODE
    "
      #include <immintrin.h>
      int main() {
        __m256 a;
        a = _mm256_set1_ps(0);
        return 0;
      }
    ")

set(AVX512_CODE
    "
    #include <immintrin.h>
    int main() {
      __m512 a = _mm512_set1_epi32(10);
      __m512 b = _mm512_set1_epi32(20);

      __m512 result = _mm512_add_epi32(a, b);
      return 0;
    }
    ")

include(CheckCXXCompilerFlag)

# Function to check compiler support and hardware capability for a given flag
function(check_compiler_and_hardware_support FLAG CODE_VAR EXTENSION_NAME)
  check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORTS_${EXTENSION_NAME})
  if(COMPILER_SUPPORTS_${EXTENSION_NAME})
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${FLAG}")

    check_cxx_source_runs("${${CODE_VAR}}"
                          SYSTEM_SUPPORTS_${EXTENSION_NAME}_EXTENSIONS)
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    if(SYSTEM_SUPPORTS_${EXTENSION_NAME}_EXTENSIONS)
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} ${FLAG}"
          PARENT_SCOPE)
      message(STATUS "Building with ${FLAG}")
    else()
      message(
        STATUS "Compiler supports ${FLAG} flag but the target machine does not "
               "support ${EXTENSION_NAME} instructions")
    endif()
  endif()
endfunction()

# Build SSE/AVX/AVX512 code only on x86-64 processors.
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
  check_compiler_and_hardware_support("-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512vnni" "AVX512_CODE" "AVX512")
  # check_compiler_and_hardware_support("-mavx512bw" "AVX512_CODE" "AVX512")
  # check_compiler_and_hardware_support("-mavx512vl" "AVX512_CODE" "AVX512")
  check_compiler_and_hardware_support("-mavx" "AVX_CODE" "AVX")

  check_cxx_compiler_flag("-msse" CXX_SSE)
  if(CXX_SSE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    message(STATUS "Building with SSE")
  endif()
endif()