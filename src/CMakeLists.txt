project(simple-gbdt)

string(COMPARE EQUAL "${CMAKE_OSX_SYSROOT}" "iphoneos" is_ios)

set(SIMPLE_GBDT_SRCS
  gbdt.cpp
  gbdt.h
#  ml_data.cpp
  ml_data.h
  tree.h
  types.h
)

# This module defines
# TBB_INCLUDE_DIRS, where to find task_scheduler_init.h, etc.
# TBB_LIBRARY_DIRS, where to find libtbb, libtbbmalloc
# TBB_INSTALL_DIR, the base TBB install directory
# TBB_LIBRARIES, the libraries to link against to use TBB.
# TBB_DEBUG_LIBRARIES, the libraries to link against to use TBB with debug symbols.
# TBB_FOUND, If false, don't try to use TBB.

# if(NOT is_ios AND NOT is_android)
#   find_package(TBB)
#   if( ${TBB_FOUND} )
#     message("found TBB ${TBB_LIBRARIES} ${TBB_LIBRARY_DIRS} ${TBB_INCLUDE_DIRS}")
#     add_definitions(-DHAS_TBB=1)
#   else()
#     set(TBB_LIBRARIES "") ## Empty
#   endif()
# endif()

find_package(Boost CONFIG REQUIRED filesystem system serialization iostreams )

add_library(simple-gbdt ${SIMPLE_GBDT_SRCS})
target_link_libraries(simple-gbdt ${TBB_LIBRARIES} Boost::filesystem Boost::system Boost::serialization Boost::iostreams) 