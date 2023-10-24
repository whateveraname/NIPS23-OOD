cmake_minimum_required(VERSION 2.8)

project(final)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP support")
endif()

add_subdirectory(faiss)

set(ENV{MKLROOT} "/opt/intel/oneapi/mkl/latest")
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
message(STATUS "${MKL_IMPORTED_TARGETS}")

# add_compile_options(-fsanitize=address) 
# add_link_options(-fsanitize=address)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO -Wno-unused-variable -Wno-unused-function -Wno-reorder)

# add_executable(search cpp/search.cpp)
# target_link_libraries(search faiss MKL::MKL)

add_subdirectory(pybind11)
pybind11_add_module(graphood python/graphood.cpp)
target_link_libraries(graphood faiss MKL::MKL)