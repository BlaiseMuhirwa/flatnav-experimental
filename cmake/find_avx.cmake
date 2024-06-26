cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  EXEC_PROGRAM(cat ARGS "/proc/cpuinfo" OUTPUT_VARIABLE CPUINFO)

  STRING(REGEX MATCH ".*\\ssse.*" SSE_FOUND ${CPUINFO})
  STRING(REGEX MATCH ".*\\ssse3.*" SSE3_FOUND ${CPUINFO})
  STRING(REGEX MATCH ".*\\avx.*" AVX_FOUND ${CPUINFO})
  STRING(REGEX MATCH ".*\\avx2.*" AVX2_FOUND ${CPUINFO})
  STRING(REGEX MATCH ".*\\avx512.*" AVX512_FOUND ${CPUINFO})

  if(SSE_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    message(STATUS "Building with SSE")
  endif()

  if(SSE3_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
    message(STATUS "Building with SSE3")
  endif()

  if(AVX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    message(STATUS "Building with AVX")
  endif()

  if(AVX2_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    message(STATUS "Building with AVX2")
  endif()

  if(AVX512_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
    message(STATUS "Building with AVX512")
  endif()

elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  EXEC_PROGRAM("/usr/sbin/sysctl -n machdep.cpu.features" OUTPUT_VARIABLE CPUINFO)

  STRING(REGEX REPLACE "^.*[^S](SSE).*$" "\\1" SSE_FOUND ${CPUINFO})
  STRING(REGEX REPLACE "^.*[^S](SSE3).*$" "\\1" SSE3_FOUND ${CPUINFO})

  STRING(COMPARE EQUAL "SSE" ${SSE_FOUND} SSE_FOUND)
  STRING(COMPARE EQUAL "SSE3" ${SSE3_FOUND} SSE3_FOUND)

  if(SSE_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
    message(STATUS "Building with SSE")
  endif()

  if(SSE3_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
    message(STATUS "Building with SSE2")
  endif()


endif()