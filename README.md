pybtas: A python interface to BTAS
----------------------------------

We provide an interface to CP3 which takes a 3-tensor with symmetric values
under exchange of the first two tensor indices. We return the factorized
matrices which can be used in THC.


## Installation 

You will need CMake, OpenBLAS, pybind11, Boost, and Eigen3, BLAS/LAPACK C++ interfaces, which can all be installed from conda-forge via:

```bash
conda install cmake openblas pybind11 boost-cpp eigen lapackpp blaspp
```

Alternatively, one can use the `conda_env.yml` file to install the required dependencies.

Then run

```bash
pip install .
```

Then run `pytest tests` to run the functionality tests.
