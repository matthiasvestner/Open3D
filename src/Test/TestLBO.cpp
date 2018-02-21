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
#include <fstream>


std::shared_ptr<three::TriangleMesh> CreateMeshTetraeder()
{
	auto mesh_ptr = std::make_shared<three::TriangleMesh>();

	mesh_ptr->vertices_.resize(4);
	mesh_ptr->vertices_[0] = Eigen::Vector3d(1,0,0);
	mesh_ptr->vertices_[1] = Eigen::Vector3d(0,1,0);
	mesh_ptr->vertices_[2] = Eigen::Vector3d(0,0,1);
	mesh_ptr->vertices_[3] = Eigen::Vector3d(1,1,1);


	mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 1, 2));
	mesh_ptr->triangles_.push_back(Eigen::Vector3i(1, 3, 2));
	mesh_ptr->triangles_.push_back(Eigen::Vector3i(2, 3, 0));
	mesh_ptr->triangles_.push_back(Eigen::Vector3i(0, 3, 1));

	return mesh_ptr;
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

void eigendecomposition(const Eigen::SparseMatrix<double> *S,const Eigen::SparseMatrix<double> *M,int n_eigen,Eigen::VectorXd *evalues,Eigen::MatrixXd *evecs){
	Spectra::SparseSymMatProd<double> op(*S);
	Spectra::SparseCholesky<double>  Bop(*M);
	Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigs(&op, &Bop, n_eigen, n_eigen+10);
	// Spectra::SymGEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigs(&op, &Bop, n_eigen, 100);
	eigs.init();
  int nconv = eigs.compute();
  if(eigs.info() == Spectra::SUCCESSFUL)
  	*evalues = eigs.eigenvalues();
		*evecs = eigs.eigenvectors();
}






void WriteLaplacianToFile(three::TriangleMesh &mesh){
	Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "", "");

	std::ofstream fileM("M2.txt");
	if (fileM.is_open()){
		fileM << Eigen::MatrixXd(mesh.mass_matrix_).format(OctaveFmt);
	}
	std::ofstream fileS("S2.txt");
	if (fileS.is_open()){
		fileS << Eigen::MatrixXd(mesh.stiffness_matrix_).format(OctaveFmt);
	}
}


int main(int argc, char *argv[])
{
	using namespace three;

	auto mesh = CreateMeshSphere(0.05);

	// auto mesh = CreateMeshTetraeder();

	mesh->ComputeVertexNormals();
	// DrawGeometries({mesh});

	mesh->ComputeLBO();
  DrawGeometries({mesh});
	auto M = Eigen::MatrixXd(mesh->mass_matrix_);
	auto S = Eigen::MatrixXd(mesh->stiffness_matrix_);
  std::vector<double> A = mesh->triangle_areas_;
	// for (auto value : A) {
  //   std::cout << value << '\n';
	// }
	// std::cout << M << '\n';
	// std::cout << S << '\n';
	int n_eigen = 20;
	Eigen::VectorXd evalues(n_eigen);
	Eigen::MatrixXd evecs;
  WriteLaplacianToFile(*mesh);

	eigendecomposition(&(mesh->stiffness_matrix_),&(mesh->mass_matrix_),n_eigen,&evalues,&evecs);
  std::cout <<  "Evals rows: " << evalues.rows() << " Evals cols: " << evalues.cols() << '\n';
	std::cout <<  "Evecs rows: " <<  evecs.rows() << " Evecs cols: " <<  evecs.cols() << '\n';
	std::vector<double> phi(mesh->vertices_.size());
	for (int j=0;j<evecs.cols();j++){
		for (int i=0;i<mesh->vertices_.size();i++){
			phi[i] = evecs(i,j);
		}
		std::cout << "Eigenfunction No." << j << " Energy: " << evalues(j) << '\n';
	  ScalarMapToColors(*mesh,phi);
	  DrawGeometries({mesh});
	}


}
