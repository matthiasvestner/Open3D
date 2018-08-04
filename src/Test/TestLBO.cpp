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


	mesh_ptr->triangles_.resize(4);
	mesh_ptr->triangles_[0] = Eigen::Vector3i(0, 1, 2);
	mesh_ptr->triangles_[1] = Eigen::Vector3i(1, 3, 2);
	mesh_ptr->triangles_[2] = Eigen::Vector3i(2, 3, 0);
	mesh_ptr->triangles_[3] = Eigen::Vector3i(0, 3, 1);

	return mesh_ptr;
}

void ScalarMapToColors(three::TriangleMesh &mesh, const std::vector<double> &f, double a, double b)
{
	auto cmap = three::ColorMapJet();
	mesh.vertex_colors_.resize(mesh.vertices_.size());
	// std::cout << f[3] << '\n';
	auto a2 = *std::min_element(std::begin(f),std::end(f));
	auto b2 = *std::max_element(std::begin(f),std::end(f));
	a = std::min(a,a2);
	b = std::min(b,b2);
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

void test1()
{
	using namespace three;

	std::cout << "The discrete Laplacian operator (LBO) for a triangular Mesh with n vertices is the nxn-matrix L=M^{-1}S" << '\n';
	std::cout << "The nxn-matrix M is called MASSMATRIX and is symmetric, positive definit and sparse." << '\n';
	std::cout << "Its support is on the diagonal and on the edges of the mesh, ie. M_{ij}!=0 iff v_i and v_j share an edge." << '\n';
	std::cout << "In that case the entry M_{ij} is given by 1/12th the sum of areas of the triangles adjecent to that edge (v_i,v_j)." << '\n';
	std::cout << "The diagonal entries M_{ii} are given by 1/6th the sum of areas of all triangles having v_i as one of its vertices." << '\n';
  std::cout << "As a consequence the diagonal entries are the sum of the off-diagonal entries of the corresponding row." << '\n';

	auto tetraeder = CreateMeshTetraeder();
	tetraeder->ComputeLBO();

	auto M = Eigen::MatrixXd(tetraeder->mass_matrix_);
	auto S = Eigen::MatrixXd(tetraeder->stiffness_matrix_);

	std::cout << "As a first example we consider the tetraeder with vertices (1,0,0),(0,1,0),(0,0,1),(1,1,1)" << '\n';
	std::cout << "Its four faces are equiliteral triangles with edge length sqrt(2) and areas sqrt(3)/2 (0.866026)." << '\n';
	std::cout << "The mass matrix is thus given by" << '\n';

	std::cout << M << '\n';

	std::cout << "Notice the lack of sparsity since in this simple example all vertices are connected via edges." << '\n';
	std::cout << "The STIFFNESS MATRIX S is also symmetric and has the same sparsity pattern as M." << '\n';
	std::cout << "Its off-diagonal entries S_{ij} are given by -(cot(alpha)+cot(beta))/2 where alpha and beta are the angles opposing the edge (v_i,v_j)." << '\n';
	std::cout << "The diagonal entries S_{ii} are given by the negative sum of all cotangens of angles of triangles having v_i as one of its vertices except for the angles at v_i." << 'n';
	std::cout << "As a consequence all rows sum up to 0." << '\n';
	std::cout << "In the case of the tetraeder all angles are pi/3 with cotangens 0.57735 and the stiffness matrix is thus given by" << '\n';

	std::cout << S << '\n';

}

void eigendecomposition(const Eigen::SparseMatrix<double> *S,const Eigen::SparseMatrix<double> *M,int n_eigen){
	Spectra::SparseSymMatProd<double> op(*S);
	Spectra::SparseCholesky<double>  Bop(*M);



	Spectra::SymGEigsSolver<double, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEIGS_CHOLESKY> eigs(&op, &Bop, n_eigen, 500);
	eigs.init();
  eigs.compute();
	Eigen::VectorXd evalues;
  Eigen::MatrixXd evecs;
	std::cout << eigs.info() << '\n';
	{
		std::cout << "Successful" << '\n';
  	evalues = eigs.eigenvalues();
		evecs = eigs.eigenvectors();
		std::cout << "Eigenvalues of the sphere:" << '\n';
		std::cout << evalues << '\n';
	}

}


void test2()
{
	using namespace three;


	std::cout << "The LBO L=M^{-1}S is self adjoint wrt. the inner product induced by the spd matrix M, ie <Lx,y>_M = <x,Ly>_M" << '\n';
	std::cout << "This is a direct consequence of the symmetry of S." << '\n';
	std::cout << "The eigenvalue problem lx=Lx is equivalent to the generalized eigenvalue problem lMx=Sx " << '\n';
	std::cout << "As a consequence the eigenvalues l are real, in fact they are all non-positive and can be sorted such that they decrease approximately linearly (with slant inversely propoertional to the shapes area)." << '\n';
	std::cout << "In case of meshes without boundary exactly one eigenvalue equals 0." << '\n';
	std::cout << "The multiplicity of eigenvalues has a connection to the symmetry of the shape." << '\n';


	std::cout << "The eigenvalues correspond to the Dirichlet energy of the corresponding eigenfunctions, which can be chosen to be orthonormal wrt. the inner product induced by M." << '\n';

	auto sphere = CreateMeshSphere();
	sphere->ComputeLBO();
	int n_eigen = 10;

	Eigen::VectorXd evalues;
	Eigen::MatrixXd evecs;

	eigendecomposition(&(sphere->stiffness_matrix_),&(sphere->mass_matrix_),n_eigen);

}




int main(int argc, char *argv[])
{
	using namespace three;
	test1();
	test2();
    std::cout<< "aha\n";
	// std::cout << "First test: Tetraeder" << '\n';
	// std::cout << "4 vertices, 4 faces" << '\n';
	// std::cout << "All angles pi/3 wit cotangens 0.5774" << '\n';
  //
  //
	// auto tetraeder = CreateMeshTetraeder();
	// tetraeder->ComputeLBO();
	// DrawGeometries({tetraeder});
  // tetraeder->ComputeLBO();
  //
  //
  //
  //
	// auto mesh = CreateMeshSphere(0.05);
  //
  //
	// mesh->ComputeVertexNormals();
	// // DrawGeometries({mesh});
  //
	// mesh->ComputeLBO();
  // DrawGeometries({mesh});
	// auto M = Eigen::MatrixXd(mesh->mass_matrix_);
	// auto S = Eigen::MatrixXd(mesh->stiffness_matrix_);
  // std::vector<double> A = mesh->triangle_areas_;
	// // for (auto value : A) {
  // //   std::cout << value << '\n';
	// // }
	// // std::cout << M << '\n';
	// // std::cout << S << '\n';
	// int n_eigen = 20;
	// Eigen::VectorXd evalues(n_eigen);
	// Eigen::MatrixXd evecs;
  // WriteLaplacianToFile(*mesh);
  //
	// eigendecomposition(&(mesh->stiffness_matrix_),&(mesh->mass_matrix_),n_eigen,&evalues,&evecs);
  // std::cout <<  "Evals rows: " << evalues.rows() << " Evals cols: " << evalues.cols() << '\n';
	// std::cout <<  "Evecs rows: " <<  evecs.rows() << " Evecs cols: " <<  evecs.cols() << '\n';
	// std::vector<double> phi(mesh->vertices_.size());
	// for (int j=0;j<evecs.cols();j++){
	// 	for (int i=0;i<mesh->vertices_.size();i++){
	// 		phi[i] = evecs(i,j);
	// 	}
	// 	std::cout << "Eigenfunction No." << j << " Energy: " << evalues(j) << '\n';
	//   ScalarMapToColors(*mesh,phi);
	//   DrawGeometries({mesh});
	// }


}
