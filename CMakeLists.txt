cmake_minimum_required(VERSION 3.19)
project(CPZLIB VERSION 0.1.0
               DESCRIPTION "Efficient Constrained Polynomial Zonotopes for C++"
               LANGUAGES CXX)

# CMake modules
include(FetchContent)

# Setup CCache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Dependencies
FetchContent_Declare

set(HEADER_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cpzlib.hh")

# Test executable
add_executable(cpzlib_test test/cpzlib_test.cc)
target_compile_features(cpzlib_test PUBLIC cxx_std_20)
set_target_properties(cpzlib_test PROPERTIES
                                  CXX_EXTENSIONS OFF
                                  CXX_STANDARD_REQUIRED YES
                                  LINK_WHAT_YOU_USE ON)

# Benchmark executable
add_executable(cpzlib_benchmark test/cpzlib_benchmarks.cc)
target_compile_features(cpzlib_benchmark PUBLIC cxx_std_20)
set_target_properties(cpzlib_benchmark PROPERTIES
                                       CXX_EXTENSIONS OFF
                                       CXX_STANDARD_REQUIRED YES
                                       LINK_WHAT_YOU_USE ON)

# Main library
add_library(cpzlib INTERFACE)
target_compile_features(cpzlib PUBLIC cxx_std_20)
set_target_properties(cpzlib PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD_REQUIRED YES)
target_sources(cpzlib INTERFACE "$<BUILD_INTERFACE:${HEADER_FILES}>")
target_include_directories(cpzlib INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>")
target_include_directories(cpzlib SYSTEM INTERFACE "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>")
