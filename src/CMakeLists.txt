# 
#========================================================
# Copyright (c) RVVLASparse and Lobachevsky State University of 
# Nizhny Novgorod and its affiliates. All rights reserved.
# 
# Copyright 2026 The RVVLASparse Authors (Evgeny Kozinov)
#
# Distributed under the MIT License
# (See file LICENSE in the root directory of this 
# source tree)
#========================================================
#

if(BUILD_SCALAR)
  add_subdirectory(scalar)
endif()
  
if(BUILD_RVV)
  add_subdirectory(rvv)
endif()