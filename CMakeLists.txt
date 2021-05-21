cmake_minimum_required(VERSION 3.19)
project(CPZLIB VERSION 0.1.0
               DESCRIPTION "Efficient Constrained Polynomial Zonotopes for C++"
               LANGUAGES CXX)

# CMake modules
include(FetchContent)
include(cmake/CPM.cmake)
set(CPM_DOWNLOAD_ALL NO)
set(CPM_USE_LOCAL_PACKAGES YES)

# Setup CCache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Dependencies
# eigen
CPMFindPackage(
  NAME Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.bz2
  DOWNLOAD_ONLY YES)

if(Eigen3_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
elseif(Eigen3_FOUND)
  add_library(Eigen ALIAS Eigen3)
endif()


# robin_hood
CPMFindPackage(
  NAME robin-hood-hashing
  GITHUB_REPOSITORY martinus/robin-hood-hashing
  GIT_TAG master)

# pdqsort
CPMFindPackage(
  NAME pdqsort
  GITHUB_REPOSITORY orlp/pdqsort
  GIT_TAG master
  DOWNLOAD_ONLY YES)

if(pdqsort_ADDED)
  add_library(pdqsort INTERFACE IMPORTED)
  target_include_directories(pdqsort INTERFACE ${pdqsort_SOURCE_DIR})
endif()

# dbg
CPMFindPackage(
  NAME dbg-macro
  GITHUB_REPOSITORY sharkdp/dbg-macro
  GIT_TAG master
  DOWNLOAD_ONLY YES)

if(dbg-macro_ADDED)
  add_library(dbg-macro INTERFACE IMPORTED)
  target_include_directories(dbg-macro INTERFACE ${dbg-macro_SOURCE_DIR})
endif()

# celero
# CPMFindPackage(
#   NAME celero
#   GITHUB_REPOSITORY DigitalInBlue/Celero
#   GIT_TAG master)

# rapidcheck
CPMFindPackage(
  NAME rapidcheck
  GITHUB_REPOSITORY emil-e/rapidcheck
  GIT_TAG master)

set(HEADER_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/cpzlib.hh")

# Main library
set(MAIN_DEPENDENCIES Eigen robin-hood-hashing pdqsort)
add_library(cpzlib INTERFACE)
target_compile_features(cpzlib INTERFACE cxx_std_20)
set_target_properties(cpzlib PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD_REQUIRED YES)
target_sources(cpzlib INTERFACE "$<BUILD_INTERFACE:${HEADER_FILES}>")
target_include_directories(cpzlib INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>")
target_include_directories(cpzlib SYSTEM INTERFACE "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>")
target_link_libraries(cpzlib PUBLIC INTERFACE ${MAIN_DEPENDENCIES})

# Test executable
# TODO: Do this the "right" way with enable_tests etc.
add_executable(cpzlib_test test/cpzlib_test.cc)
target_compile_features(cpzlib_test PUBLIC cxx_std_20)
set_target_properties(
  cpzlib_test PROPERTIES
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED YES
  LINK_WHAT_YOU_USE ON)
target_link_libraries(cpzlib_test PUBLIC INTERFACE cpzlib)

# Benchmark executable
# add_executable(cpzlib_benchmark test/cpzlib_benchmarks.cc)
# target_compile_features(cpzlib_benchmark PUBLIC cxx_std_20)
# set_target_properties(
#   cpzlib_benchmark PROPERTIES
#   CXX_EXTENSIONS OFF
#   CXX_STANDARD_REQUIRED YES
#   LINK_WHAT_YOU_USE ON)
# target_link_libraries(cpzlib_benchmark PUBLIC INTERFACE cpzlib)
# target_link_libraries(cpzlib_benchmark PUBLIC celero)
