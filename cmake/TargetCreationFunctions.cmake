MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${curdir}/${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

function(create_project_lib TARGET )
    file(GLOB_RECURSE TARGET_SRC "*.c*")
    file(GLOB_RECURSE TARGET_HD "*.h*")
    add_library(${TARGET} STATIC ${TARGET_SRC} ${TARGET_HD})

    target_include_directories(${TARGET} PUBLIC "${CMAKE_SOURCE_DIR}/include")

    get_property ( LIB_LIST GLOBAL PROPERTY LIBS_P)
    list(APPEND LIB_LIST ${TARGET})
    set_property ( GLOBAL PROPERTY LIBS_P ${LIB_LIST})
endfunction()

function(create_executable_project TARGET)
    file(GLOB_RECURSE TARGET_SRC "*.c*")
    file(GLOB_RECURSE TARGET_HD "*.h*")
    add_executable(${TARGET} ${TARGET_SRC} ${TARGET_HD})

    target_include_directories(${TARGET} PUBLIC "${CMAKE_SOURCE_DIR}/include")

    get_property ( LIB_LIST GLOBAL PROPERTY LIBS_P)
    target_link_libraries(${TARGET} ${LIB_LIST})
endfunction()