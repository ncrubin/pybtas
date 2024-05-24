cmake_minimum_required(VERSION 3.24...3.29)
set(CMAKE_CXX_STANDARD 17)
project(pybtas)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# use FetchContent to get BTAS source and targets
include(FetchContent)
FetchContent_Declare(BTAS
  GIT_REPOSITORY https://github.com/valeevgroup/btas.git
  GIT_TAG        master
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(BTAS)

pybind11_add_module(_pybtas src/pybtas/main.cpp)

find_package(BTAS REQUIRED)
find_package(Eigen3 REQUIRED)
target_link_libraries(_pybtas PRIVATE Eigen3::Eigen BTAS)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_pybtas PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

install(TARGETS _pybtas DESTINATION .)