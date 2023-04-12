cmake_minimum_required(VERSION 3.19)
project(ExampleExperiment
  DESCRIPTION "Example COCO experiment"
  LANGUAGES CXX C)

find_library(MATH_LIBRARY m)


add_library(coco STATIC coco.c coco.h)
target_include_directories(coco PUBLIC .)
if(MATH_LIBRARY)
    target_link_libraries(coco PUBLIC ${MATH_LIBRARY})
endif()
# Add warning flags
if (MSVC)
  target_compile_options(coco PRIVATE "/W3")
elseif (CC_HAS_WALL_ETC)
  target_compile_options(coco PRIVATE -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion)
endif()

find_package(fmt REQUIRED)
find_package(nlshade REQUIRED)


add_executable(example_experiment example_experiment.cc)
target_compile_options(example_experiment PRIVATE  -g -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion)
target_link_libraries(example_experiment PUBLIC coco nlshade::nlshade)
set_property(TARGET example_experiment PROPERTY CXX_STANDARD 17)
if(MATH_LIBRARY)
    target_link_libraries(example_experiment PUBLIC ${MATH_LIBRARY})
endif()
