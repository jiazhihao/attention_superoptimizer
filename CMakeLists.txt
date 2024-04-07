cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(ASO LANGUAGES C CXX CUDA)

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

set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CUTLASS_NVCC_ARCHS "80;86")

file(GLOB_RECURSE ASO_SRCS
  src/*.cc
)

file(GLOB_RECURSE ASO_CUDA_SRCS
  src/*.cu
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

#set CUDA version
include(deps/cutlass/CUDA.cmake)
string(REPLACE "." ";" CUDA_VERSION_PARTS ${CUDA_VERSION})
list(GET CUDA_VERSION_PARTS 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_PARTS 1 CUDA_VERSION_MINOR)
list(APPEND CMAKE_CUDA_FLAGS "-D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR} -DCUTLASS_ARCH_MMA_SM80_ENABLED")

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
    list(APPEND ASO_SRCS ${ASO_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    list(APPEND ASO_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND ASO_LINK_LIBS ${CUDA_CUDA_LIBRARY})
    list(APPEND ASO_LINK_LIBS ${CUDA_CUDNN_LIBRARY})
    list(APPEND ASO_LINK_LIBS ${CUDA_CUBLAS_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

#set CUTLASS
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/deps/cutlass)
include_directories(deps/cutlass/include)
include_directories(deps/cutlass/tools/util/include)

#include_directories(${ASO_INCLUDE_DIRS})

#set Z3
find_package(Z3 REQUIRED)
message(STATUS "Z3_FOUND: ${Z3_FOUND}")
message(STATUS "Found Z3 ${Z3_VERSION_STRING}")
message(STATUS "Z3_DIR: ${Z3_DIR}")
include_directories(${Z3_CXX_INCLUDE_DIRS})
#target_link_libraries(aso_runtime PUBLIC ${Z3_LIBRARIES})
list(APPEND ASO_LINK_LIBS ${Z3_LIBRARIES})

add_subdirectory(deps/json)
list(APPEND ASO_LINK_LIBS nlohmann_json::nlohmann_json)

#add_subdirectory(deps/z3 z3 EXCLUDE_FROM_ALL)
#include_directories(deps/z3/src/api/c++)
#include_directories(deps/z3/src/api)
#list(APPEND ASO_LINK_LIBS z3)

add_library(aso_runtime ${ASO_SRCS})

set_target_properties(aso_runtime
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(aso_runtime PROPERTIES CUDA_ARCHITECTURES "86")


target_compile_features(aso_runtime PUBLIC cxx_std_17)

target_link_libraries(aso_runtime ${ASO_LINK_LIBS})

target_include_directories(aso_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

install(TARGETS aso_runtime
    LIBRARY DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN dnn)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

# Multi-Query Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

# Group-Query Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

# Multi-Head Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

# LoRA
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN lora)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN search_test)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN from_checkpoint)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN profile)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN layout_optimize)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} aso_runtime)
endif()

option(ASO_BUILD_UNIT_TEST "build unit tests" OFF)

# GTest
if (ASO_BUILD_UNIT_TEST)
  enable_testing()
  find_package(GTest)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)
  endif()
  include_directories(${GTEST_INCLUDE_DIRS})
  add_subdirectory(tests)
endif()
