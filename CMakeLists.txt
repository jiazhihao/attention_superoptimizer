cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(ASO LANGUAGES CXX CUDA)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

add_subdirectory(deps/cutlass)

set(CUTLASS_NVCC_ARCHS 80)

file(GLOB_RECURSE ASO_SRCS
  src/*.cu
  src/*.cc
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(cmake/FindCUDA.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    list(APPEND TASO_SRCS ${TASO_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDNN)
    list(APPEND TASO_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND TASO_LINK_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND TASO_LINK_LIBS ${CUDA_CUDNN_LIBRARY})
    list(APPEND TASO_LINK_LIBS ${CUDA_CUBLAS_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

include_directories(deps/cutlass/include)
include_directories(deps/cutlass/tools/util/include)
include_directories(${ASO_INCLUDE_DIRS})

add_executable(aso ${ASO_SRCS})

# target_link_libraries(aso ${ASO_EXT_LIBRARIES})
