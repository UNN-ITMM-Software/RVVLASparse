cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(SparseMatrix)

set (CMAKE_CONFIGURATION_TYPES "Debug;Release") 
set (CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenMP support for C++
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

option(WITH_SCALAR_LIB     "Build the scalar library" ON)
option(WITH_RVV_LIB        "Build the RVV library" ON)

include(cmake/TargetCreationFunctions.cmake)
include(cmake/TestEnvironment.cmake)
include(cmake/OutputFlagsOption.cmake)

if(BUILD_SCALAR)
  add_compile_options(-DSP_M_SCALAR)
endif()

if(BUILD_RVV)
  if(HAVE_RVV_0P7)
    add_compile_options(-DSP_M_RVV_0P7)
  endif()
  if(HAVE_RVV_1P0)
    add_compile_options(-DSP_M_RVV_1P0)
  endif()
endif()

if(HAVE_MKL)
  add_compile_options(-DSP_M_MKL)
endif()

add_subdirectory(src)