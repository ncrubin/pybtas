#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>


#ifdef BTAS_HAS_BOOST_SERIALIZATION
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/complex.hpp>
#endif  // BTAS_HAS_BOOST_SERIALIZATION

#include "btas/tensor.h"
#include "btas/btas.h"
#include "btas/generic/converge_class.h"
#include "btas/tensor_func.h"
#include "btas/generic/contract.h"


using btas::Range;
using btas::Tensor;

using DTensor = Tensor<double>;
using conv_class = btas::FitCheck<DTensor>;
using btas::CP_ALS;

namespace py = pybind11;

std::vector<Eigen::MatrixXd> cp3(py::array_t<double> inArray, int rank, double conv_eps=1.0e-4) {
    // request a buffer descriptor from Python
    py::buffer_info buffer_info = inArray.request();

    // extract data an shape of input array
    double *data = static_cast<double *>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    // wrap ndarray in Eigen::Map:
    // the second template argument is the rank of the tensor and has to be known at compile time
    Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> in_tensor(data, shape[0], shape[1], shape[2]);

    DTensor T3(shape[0], shape[1], shape[2]);
    T3.fill(0.);

    for (int i=0; i < shape[0]; i++) {
    for (int j=0; j < shape[1]; j++) {
    for (int k=0; k < shape[2]; k++) {
      T3(i, j, k) = in_tensor(i, j, k);
      //std::cout << "(" << i << ", " << j << ", " << k << ") " << T3(i, j, k) << std::endl;
    }
    }
    }

    double t3norm = sqrt(dot(T3, T3));
    conv_class conv(conv_eps);
    std::vector<size_t> symm = {0, 0, 2};
    CP_ALS<DTensor, conv_class> A1(T3, symm);
    conv.set_norm(t3norm);
    //std::cout << "Norm " << t3norm << std::endl;
    //std::cout << "Calling facotrization" << std::endl;
    double diff = A1.compute_rank(rank, conv);
    auto factors = A1.get_factor_matrices();

    //std::cout << "\nfac dims" << std::endl;
    //for (auto fac : factors){
    //  if (fac.rank() == 2) {
    //  std::cout << "Rank of tensor is 2 " << std::endl;
    //  std::cout << fac.extent(0) << ", " << fac.extent(1) << std::endl;
    //  for (int i =0; i  < fac.extent(0); ++i){
    //    for(int j = 0; j < fac.extent(1); ++j){
    //      std::cout  << "(" << i << ", " << j << ")\t" << fac(i, j) << std::endl;
    //    }
    //  }
    //  std::cout << std::endl;
    //} else if (fac.rank() == 1) {
    //  std::cout << "Rank of tensor is 1 " << std::endl;
    //  std::cout << fac.extent(0) << std::endl;
    //  for(int i=0; i < fac.extent(0); ++i){
    //    std::cout << "(" << i << ") " << fac(i) << std::endl;
    //  }
    //}
    //}

    // auto reconstructed_tensor = A1.reconstruct();

    // for (int i=0; i < shape[0]; i++) {
    // for (int j=0; j < shape[1]; j++) {
    // for (int k=0; k < shape[2]; k++) {
    //   std::cout << "(" << i << ", " << j << ", " << k << ") " << T3(i, j, k) - reconstructed_tensor(i, j, k)  << std::endl;
    // }
    // }
    // }



    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(factors[0].extent(0), factors[0].extent(1));
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(factors[2].extent(0), factors[2].extent(1));

    Eigen::MatrixXd scale_vec = Eigen::MatrixXd::Zero(factors[3].extent(0), 1);

    //std::cout << "PRINT BETA MAT" << std::endl;
    //for(int i = 0; i < factors[0].extent(0); i++){
    //  for(int j = 0; j < factors[0].extent(1); j++){
    //    beta(i, j) = factors[0](i, j);
    //    std::cout << "(" << i << ", " << j << ") " << beta(i, j) << std::endl;
    //  }
    //}
    //for(int i = 0; i < factors[2].extent(0); i++){
    //  for(int j = 0; j < factors[2].extent(1); j++){
    //    gamma(i, j) = factors[2](i, j);
    //  }
    //}
    //std::cout << "scalar vector size " << factors[3].extent(0) << std::endl;
    //for(int i = 0; i < factors[3].extent(0); i++){
    //    scale_vec(i, 0) = factors[3](i);
    //    std::cout << factors[3](i) << std::endl;
    //}

    std::vector<Eigen::MatrixXd> matrices;
    matrices.push_back(beta);
    matrices.push_back(gamma);
    matrices.push_back(scale_vec);
    //std::cout << "Returning"<< std::endl;
    return matrices;

}


PYBIND11_MODULE(pybtas, m) {
    m.def("cp3", &cp3, py::return_value_policy::move,
        py::arg("inArray"), py::arg("rank"), py::arg("conv_eps")=1.0e-4);
}
