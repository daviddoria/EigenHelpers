cmake_minimum_required(VERSION 2.6)

PROJECT(EigenHelpers)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/" ${CMAKE_MODULE_PATH})

if(NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  set(EigenHelpers_includes ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
endif()

include_directories(/media/portable/src/Eigen)
FIND_PACKAGE(Eigen3 REQUIRED) #requires FindEigen3.cmake to be in the source directory
include_directories(${EIGEN3_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}/unsupported/)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET")

add_library(EigenHelpers EigenHelpers.cpp)

# This is needed for shared_ptr and the trick using enable_if and if_fundamental to allow scalars
# to be treated as the 0th component of a vector.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

SET(EigenHelpers_BuildTests OFF CACHE BOOL "EigenHelpers_BuildTests")
if(EigenHelpers_BuildTests)
  add_subdirectory(Tests)
endif()
