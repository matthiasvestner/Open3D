// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <iostream>

#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>
#include <Visualization/Utility/ColorMap.h>
// #include <Eigen/Eigenvalues>
#include <SymEigsSolver.h>
#include <SymGEigsSolver.h>
#include <MatOp/DenseSymMatProd.h>
#include <MatOp/SparseCholesky.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>


void eigendecomposition(Eigen::SparseMatrix<double> S,Eigen::SparseMatrix<double> M,int n_eigen,Eigen::VectorXd *evalues,Eigen::MatrixXd *evecs){
	Spectra::SparseSymMatProd<double> op(S);
	Spectra::SparseCholesky<double>  Bop(M);
	// Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigs(&op, &Bop, n_eigen, n_eigen+10);
	Spectra::SymGEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigs(&op, &Bop, n_eigen, n_eigen+1);
	eigs.init();
  int nconv = eigs.compute();
  if(eigs.info() == Spectra::SUCCESSFUL)
  	*evalues = eigs.eigenvalues();
		*evecs = eigs.eigenvectors();
}

void ScalarMapToColors(three::TriangleMesh &mesh, const std::vector<double> &f)
{
	auto cmap = three::ColorMapJet();
	mesh.vertex_colors_.resize(mesh.vertices_.size());
	// std::cout << f[3] << '\n';
	auto a = *std::min_element(std::begin(f),std::end(f));
	auto b = *std::max_element(std::begin(f),std::end(f));
	// std::cout << a << '\n';
	// std::cout << b << '\n';
	for (size_t i=0;i<mesh.vertices_.size();i++){
		mesh.vertex_colors_[i] = cmap.GetColor((f[i]-a)/(b-a));
	}
}

void PaintMesh(three::TriangleMesh &mesh, const Eigen::Vector3d &color)
{
	mesh.vertex_colors_.resize(mesh.vertices_.size());
	for (size_t i = 0; i < mesh.vertices_.size(); i++) {
		mesh.vertex_colors_[i] = color;
	}
}

int main(int argc, char *argv[])
{
	using namespace three;

	auto mesh = CreateMeshSphere(0.05);
	// auto mesh = CreateMeshFromFile("/Users/mvestner/ownCloud/Documents/Promotion/registrations/tr_reg_000.ply");
	mesh->ComputeVertexNormals();
	mesh->ComputeLBO();
	int n_eigen = 20;
	Eigen::VectorXd evalues(n_eigen);
	Eigen::MatrixXd evecs;

	eigendecomposition(mesh->stiffness_matrix_,mesh->mass_matrix_,n_eigen,&evalues,&evecs);
	std::cout <<  "Evals rows: " << evalues.rows() << " Evals cols: " << evalues.cols() << '\n';
	std::cout <<  "Evecs rows: " <<  evecs.rows() << " Evecs cols: " <<  evecs.cols() << '\n';
	//
	// std::cout << evalues << std::endl;
	std::vector<double> phi(mesh->vertices_.size());
	for (int j=0;j<evecs.cols();j++){
		std::cout << j << '\n';
		for (int i=0;i<mesh->vertices_.size();i++){
			phi[i] = evecs(i,j);
		}
		std::cout << "Eigenfunction No." << j << " Energy: " << evalues(j) << '\n';
	  ScalarMapToColors(*mesh,phi);
	  DrawGeometries({mesh});
	}


}
