cmake_minimum_required(VERSION 3.4...3.18)
project(pybtas)

add_subdirectory(pybind11)
pybind11_add_module(pybtas src/main.cpp)

include_directories(~/opt/eigen-3.3.9)

find_package(BTAS
        PATHS ~/opt/btas/)
target_link_libraries(pybtas PUBLIC BTAS::BTAS)


# add_compile_definitions(BTAS_HAS_BLAS_LAPACK=1 LAPACK_COMPLEX_CPP=1 BTAS_HAS_BOOST_SERIALIZATION=1)
# include_directories(~/opt/btas/include ~/opt/boost_1_76_0/)
# target_link_directories(numpy3tensor PUBLIC  ~/opt/btas/lib)
# target_link_libraries(numpy3tensor PUBLIC blaspp lapackpp)



# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pybtas PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
