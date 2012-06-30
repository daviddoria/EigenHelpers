# Allow headers in tests to be included like
# #include "EigenHelpers.h" rather than needing
# #include "EigenHelpers/EigenHelpers.h"
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(TestEigenHelpers TestEigenHelpers.cpp)
#target_link_libraries(TestEigenHelpers EigenHelpers) # not necessary for a template library
