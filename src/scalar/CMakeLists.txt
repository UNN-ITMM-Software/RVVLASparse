create_project_lib("spMat_${CMAKE_SYSTEM_PROCESSOR}_scalar")
set_property(TARGET "spMat_${CMAKE_SYSTEM_PROCESSOR}_scalar" PROPERTY POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options("spMat_${CMAKE_SYSTEM_PROCESSOR}_scalar" PRIVATE -fno-tree-vectorize)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options("spMat_${CMAKE_SYSTEM_PROCESSOR}_scalar" PRIVATE -fno-vectorize -fno-slp-vectorize)
endif()
