if(BUILD_SCALAR)
  add_subdirectory(scalar)
endif()
  
if(BUILD_RVV)
  add_subdirectory(rvv)
endif()