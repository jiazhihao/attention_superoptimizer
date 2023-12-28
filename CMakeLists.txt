cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(ASO LANGUAGES CXX CUDA)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}) # to include protobuf headeer files
include_directories("include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

#add_subdirectory(deps/cutlass)

set(CUTLASS_NVCC_ARCHS 80)

file(GLOB_RECURSE ASO_SRCS
  src/*.cu
  src/*.cc
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

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
  include(cmake/cuda.cmake)
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

#include_directories(deps/cutlass/include)
#include_directories(deps/cutlass/tools/util/include)
include_directories(${ASO_INCLUDE_DIRS})

add_library(lib_aso ${ASO_SRCS})

set_target_properties(lib_aso
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(lib_aso PUBLIC cxx_std_17)

target_link_libraries(lib_aso ${ASO_LINK_LIBS})

target_include_directories(lib_aso
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

install(TARGETS lib_aso
    LIBRARY DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN dnn)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/*.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} lib_aso)
endif()
