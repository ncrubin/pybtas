pybtas: A python interface to BTAS
----------------------------------

We provide an interface to CP3 which takes a 3-tensor with symmetric values
under exchange of the first two tensor indices. We return the factorized
matrices which can be used in THC.


Installation 

you will need cmake and pybind

conda install cmake
python -m pip install pybind

1) get boost and eigen and btas and blas/lapack.  I used OpenBLAS which comes
with lapack.
2) build boost.  navigate to boost dir and type ./bootstrap.sh &&  ./b2
3) navigate to btas and use  

cmake . -DBTAS_USE_BLAS_LAPACK=ON -DCMAKE_INSTALL_PREFIX=~/opt/btas -DBTAS_BUILD_UNITTEST=TRUE -DBTAS_ASSERT_THROWS=ON DBOOST_INCLUDEDIR=/usr/local/google/home/nickrubin/opt/boost_1_76_0 -DBoost_DIR=/usr/local/google/home/nickrubin/opt/boost_1_76_0

or the equivalent for your system


then type 

make check


and installation should happen along with checks. Conclude by typing 

make install



4) navigate to pybtas directory and update CMakeLists.txt with with appropriate
paths for eigen  and BTAS

5) python setup.py clean && python setup.py install


Then run test_pybtas.py to run CP3.
