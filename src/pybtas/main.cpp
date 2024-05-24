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

std::vector<Eigen::MatrixXd> cp3(py::array_t<double> inArray, int rank, double conv_eps=1.0e-4, bool random_start=false) {
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
    std::cout << "\tBTAS: Tensor-norm " << t3norm << std::endl;
    std::cout << "\tBTAS: Calling factorization" << std::endl;
    if (random_start) {
      std::cout << "\tBTAS: random start\n";
      double diff = A1.compute_rank_random(rank, conv);
    } else {
      std::cout << "\tBTAS: HOSVD start\n";
      double diff = A1.compute_rank(rank, conv);
    }
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

    auto reconstructed_tensor = A1.reconstruct();

    //for (int i=0; i < shape[0]; i++) {
    //for (int j=0; j < shape[1]; j++) {
    //for (int k=0; k < shape[2]; k++) {
    //  std::cout << "(" << i << ", " << j << ", " << k << ") " << T3(i, j, k) - reconstructed_tensor(i, j, k)  << std::endl;
    //}
    //}
    //}



    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(factors[0].extent(0), factors[0].extent(1));
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(factors[2].extent(0), factors[2].extent(1));

    Eigen::MatrixXd scale_vec = Eigen::MatrixXd::Zero(factors[3].extent(0), 1);

    //std::cout << "PRINT BETA MAT" << std::endl;
    for(int i = 0; i < factors[0].extent(0); i++){
      for(int j = 0; j < factors[0].extent(1); j++){
        beta(i, j) = factors[0](i, j);
        //std::cout << "(" << i << ", " << j << ") " << beta(i, j) << std::endl;
      }
    }
    //std::cout << "PRINT GAMMA MAT" << std::endl;
    for(int i = 0; i < factors[2].extent(0); i++){
      for(int j = 0; j < factors[2].extent(1); j++){
        gamma(i, j) = factors[2](i, j);
      }
    }
    //std::cout << "scalar vector size " << factors[3].extent(0) << std::endl;
    for(int i = 0; i < factors[3].extent(0); i++){
        scale_vec(i, 0) = factors[3](i);
        //std::cout << factors[3](i) << std::endl;
    }

    std::vector<Eigen::MatrixXd> matrices;
    matrices.push_back(beta);
    matrices.push_back(gamma);
    matrices.push_back(scale_vec);
    std::cout << "\tBTAS RUN L2 norm " << sqrt(dot(T3 - reconstructed_tensor, T3 - reconstructed_tensor)) << std::endl;
    std::cout << std::endl;
    return matrices;

}


std::vector<Eigen::MatrixXd> cp3_from_cholesky(Eigen::MatrixXd inArray, int rank, double conv_eps=1.0e-4, bool random_start=false) {
    //inAarry is now an n^2 x n-chol matrix
    int numrows = inArray.rows();
    int numcols = inArray.cols();

    int norbs = sqrt(numrows);

    DTensor T3(norbs, norbs, numcols);
    T3.fill(0.);

    //loop over columns and reshape
    for(int ll = 0; ll < numcols; ll++){
      //NOTE:  reshaped_cholvec should be a symmetric matrix
      Eigen::MatrixXd cholvec = inArray.col(ll);
      Eigen::MatrixXd reshaped_cholvec = Eigen::MatrixXd::Zero(norbs, norbs);
      for(int i = 0; i < norbs; i++){
        for(int j = 0; j < norbs; j++){
          reshaped_cholvec(i, j) = cholvec(i * norbs + j);
        }
      }

      //copy to T3
      for(int i = 0; i < norbs; i++){
        for(int j = 0; j < norbs; j++){
          T3(i, j, ll)=  reshaped_cholvec(i, j);
        }
      }
    }

    for (int i=0; i < norbs; i++) {
    for (int j=0; j < norbs; j++) {
    for (int k=0; k < numcols; k++) {
      if (abs(T3(i, j, k) - T3(j, i, k)) > 1.0e-6){
        std::cout << "\tBTAS: T3 nor symmetric as expected" << std::endl;
        //std::cout << "(" << i << ", " << j << ", " << k << ") " << T3(i, j, k) << " " << T3(j, i, k) << std::endl;
      }
    }
    }
    }

    std::cout << "\tBTAS T3dim: (" <<  T3.extent(0) << ", " << T3.extent(1) << ", " << T3.extent(2) << ")\n";

    double t3norm = sqrt(dot(T3, T3));
    conv_class conv(conv_eps);
    std::vector<size_t> symm = {0, 0, 2};
    CP_ALS<DTensor, conv_class> A1(T3, symm);
    conv.set_norm(t3norm);
    std::cout << "\tBTAS STOPPING NORM : " << t3norm << std::endl;
    std::cout << "\tBTAS: Calling factorization" << std::endl;
    if (random_start) {
      std::cout << "\tBTAS: random start\n";
      double diff = A1.compute_rank_random(rank, conv);
    } else {
      std::cout << "\tBTAS: HOSVD start\n";
      double diff = A1.compute_rank(rank, conv);
    }
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

    auto reconstructed_tensor = A1.reconstruct();
    //for (int i=0; i < shape[0]; i++) {
    //for (int j=0; j < shape[1]; j++) {
    //for (int k=0; k < shape[2]; k++) {
    //  std::cout << "(" << i << ", " << j << ", " << k << ") " << T3(i, j, k) - reconstructed_tensor(i, j, k)  << std::endl;
    //}
    //}
    //}



    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(factors[0].extent(0), factors[0].extent(1));
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(factors[2].extent(0), factors[2].extent(1));

    Eigen::MatrixXd scale_vec = Eigen::MatrixXd::Zero(factors[3].extent(0), 1);

    //std::cout << "\tBTAS: PRINT BETA MAT " << "(" << factors[0].extent(0) << ", "  << factors[0].extent(1) << ")" << std::endl;
    for(int i = 0; i < factors[0].extent(0); i++){
      for(int j = 0; j < factors[0].extent(1); j++){
        beta(i, j) = factors[0](i, j);
        //std::cout << "(" << i << ", " << j << ") " << beta(i, j) << std::endl;
      }
    }
    //std::cout << "\tBTAS: PRINT GAMMA MAT" << std::endl;
    for(int i = 0; i < factors[2].extent(0); i++){
      for(int j = 0; j < factors[2].extent(1); j++){
        gamma(i, j) = factors[2](i, j);
        //std::cout << "(" << i << ", " << j << ") " << gamma(i, j) << std::endl;
      }
    }
    //std::cout << "scalar vector size " << factors[3].extent(0) << std::endl;
    for(int i = 0; i < factors[3].extent(0); i++){
        scale_vec(i, 0) = factors[3](i);
        //std::cout << factors[3](i) << std::endl;
    }

    std::vector<Eigen::MatrixXd> matrices;
    matrices.push_back(beta);
    matrices.push_back(gamma);
    matrices.push_back(scale_vec);
    std::cout << "\tBTAS RUN L2 norm " << sqrt(dot(T3 - reconstructed_tensor, T3 - reconstructed_tensor)) << std::endl;
    std::cout << std::endl;

    return matrices;

}


Eigen::MatrixXd eigen_mat_return(Eigen::MatrixXd inArray) {
    int numrows = inArray.rows();
    int numcols = inArray.cols();
    Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(numrows, numcols);
    //std::cout << "PRINT BETA MAT" << std::endl;
    for(int i = 0; i < numrows; i++){
      for(int j = 0; j < numcols; j++){
        beta(i, j) = inArray(i, j);
      }
    }
    return beta;


}


PYBIND11_MODULE(_pybtas, m) {
    m.def("cp3", &cp3, py::return_value_policy::move,
        py::arg("inArray"), py::arg("rank"), py::arg("conv_eps")=1.0e-4,
        py::arg("random_start")=false);
    m.def("cp3_from_cholesky", &cp3_from_cholesky, py::return_value_policy::move,
        py::arg("inArray"), py::arg("rank"), py::arg("conv_eps")=1.0e-4,
        py::arg("random_start")=false);
    m.def("eigen_mat_return", &eigen_mat_return, py::return_value_policy::move,
        py::arg("inArray"));
}
