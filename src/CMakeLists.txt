project(simple-gbdt)
set(SIMPLE_GBDT_SRCS
  gbdt.cpp
  gbdt.h
  ml_data.cpp
  ml_data.h
  tree.h
  types.h
)

find_package(tbb REQUIRED)
add_library(simple-gbdt ${SIMPLE_GBDT_SRCS})
target_link_libraries(simple-gbdt ${TBB_LIBRARIES})