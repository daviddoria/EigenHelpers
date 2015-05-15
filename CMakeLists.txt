cmake_minimum_required(VERSION 2.6)

PROJECT(EigenHelpers)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/" ${CMAKE_MODULE_PATH})

#### Eigen ####
# Tell CMake to also look in the source directory to find some .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# Set the CMake variable to the value of the environment variable.
# If the build is absolutely fresh, this will be the case if the CMake variable EIGEN3_INCLUDE_DIR is not set.
if(NOT EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  message("Set EIGEN3_INCLUDE_DIR to ${EIGEN3_INCLUDE_DIR} from environment variable of the same name.")
else()
  message("EIGEN3_INCLUDE_DIR is ${EIGEN3_INCLUDE_DIR}")
endif()

if(NOT EIGEN3_FOUND)
  FIND_PACKAGE(Eigen3 REQUIRED) #requires FindEigen3.cmake to be in the source directory
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

if(NOT ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  set(EigenHelpers_includes ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
  set(EigenHelpers_libraries ${EigenHelpers_libraries} EigenHelpers PARENT_SCOPE)
endif()

add_library(EigenHelpers EigenHelpers.cpp)

# This is needed for shared_ptr and the trick using enable_if and if_fundamental to allow scalars
# to be treated as the 0th component of a vector.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

SET(EigenHelpers_BuildTests OFF CACHE BOOL "EigenHelpers_BuildTests")
if(EigenHelpers_BuildTests)
  add_subdirectory(Tests)
endif()
