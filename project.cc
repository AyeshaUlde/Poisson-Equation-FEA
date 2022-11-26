#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_q.h>
 
#include <deal.II/dofs/dof_tools.h>
 
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
 
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
 
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
 
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
 
using namespace dealii;
 
 
class Boundary_values : public Function<2>
{
public:
	virtual double value (const Point <2> &p,
			const unsigned int component = 0) const override;
};

//double Boundary_values::value(const Point<2> &p, const unsigned int ) const
//{
// if you are having functional boundary condition create logic here	 
//}
 class Right_side : public Function<2>
 {
 public:
 	virtual double value (const Point <2> &p,
 			const unsigned int component = 0) const override;
 };

  double Right_side::value(const Point<2> &p, const unsigned int ) const
 {
	  return ((4*p(0)*p(0)) + (p(1)*p(1)) + (4*p(0)*p(1)));
 }
class fea
{
public:
  fea();
 
  void run();
 
 
private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
 
  Triangulation<2> mesh;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;
  AffineConstraints<double> constraints;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> K_matrix;
 
  Vector<double> d_vector;
  Vector<double> F_vector;
};
 
 
fea::fea()
  : fe(1)
  , dof_handler(mesh)
{}
 
 
 
void fea::make_grid()
{
	//Giving specifications to generate a mesh with Hypershell geometry
	const Point<2> center(0, 0);
	const double   inner_radius = 6, outer_radius = 12;
	GridGenerator::hyper_shell(
	  mesh, center, inner_radius, outer_radius, 25);

	//Dividing the mesh into 4^4 parts
	mesh.refine_global(4);

	std::ofstream out("grid.vtk");
	GridOut       grid_out;
	grid_out.write_vtk(mesh, out);

	std::cout << "Grid written to grid.vtk" << std::endl;

	std::cout << "Number of active cells: " << mesh.n_active_cells()
	          << std::endl;
}
 
 
 
 
void fea::setup_system()
{
	//Enumeration of the domain
	dof_handler.distribute_dofs(fe);
	//Tagging the sparse entries and copying it to our temporary storage
	//sparse matrix
	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);
		std::cout << "Number of degrees of freedom : " << dof_handler.n_dofs()
				  << std::endl;

	//Reinitialize the matrix and give the required size to it
	K_matrix.reinit(sparsity_pattern);
	F_vector.reinit(dof_handler.n_dofs());
	d_vector.reinit(dof_handler.n_dofs());
}
 
 
 
void fea::assemble_system()
{
	//Calling gauss quadrature formula and in argument giving the number of points
	QGauss<2> quadrature_formula(2);

	//Computing the shape function values and its gradients
	FEValues<2> fe_values(fe, quadrature_formula,
			 update_values | update_gradients | update_quadrature_points | update_JxW_values);

	//To find the size of the stiffness matrix, which is equal to the number of degree
	//of freedom associated with the element, calculate the degree of freedom per cell
	//using fe class
		const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

	//Make a matrix and rhs vector for the local computation
	FullMatrix<double> local_stiffness_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>     local_rhs_vector(dofs_per_cell);

	//Save the indices ofthe cell in the global system as:
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	//Iterate through every cell and compute Kij and Fi
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_values.reinit(cell);
		local_stiffness_matrix = 0;
		local_rhs_vector = 0;
		//Calculating the element for the local stiffness matrix
		for (const unsigned int q_index : fe_values.quadrature_point_indices())
			for (const unsigned int a : fe_values.dof_indices())
				for (const unsigned int b : fe_values.dof_indices())
					local_stiffness_matrix(a,b) +=
							(fe_values.shape_grad(a, q_index) * //grad Na(x_q)
							 fe_values.shape_grad(b, q_index) * //grad Nb(x_q)
							 fe_values.JxW(q_index));			//dx
		for (const unsigned int q_index : fe_values.quadrature_point_indices())
		{
			const auto &x_q = fe_values.quadrature_point(q_index);
			for (const unsigned int a : fe_values.dof_indices())
				local_rhs_vector(a) += (fe_values.shape_value(a, q_index) * // Na(x_q)
									Right_side().value(x_q) *										// f(x_q) = 1 constant
									fe_values.JxW(q_index));					// dx
		}
		//Getting the values of the global indices to assemble back into the global stiffness matrix
		cell->get_dof_indices(local_dof_indices);

		//Using the distribute local to global function we transfer all the values from local matrix
		//to global matrix
		constraints.distribute_local_to_global(local_stiffness_matrix, local_rhs_vector,
				local_dof_indices, K_matrix, F_vector);
	}

	//Take all the degree of freedom indices and assign to boundary values;
	std::map<types::global_dof_index, double> boundary_values;
	//Interpolate boundary values function to the essential boundary condition
	VectorTools::interpolate_boundary_values(dof_handler,
											 0,
											 ConstantFunction<2>(300),
											 boundary_values);
	//Apply the essential boundary condition
	MatrixTools::apply_boundary_values(boundary_values,
									   K_matrix,
									   d_vector,
									   F_vector);

}
 
 
 
void fea::solve()
{
	SolverControl			 solver_control(1000, 1e-12);
	SolverCG<Vector<double>> solver(solver_control);
	solver.solve(K_matrix, d_vector, F_vector, PreconditionIdentity());
}
 
 
 
void fea::output_results() const
{
	DataOut<2> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(d_vector, "solution");
	data_out.build_patches();

	std::ofstream output("solution.vtk");
	data_out.write_vtk(output);
}
 
 
 
void fea::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}
 
 
 
int main()
{
  deallog.depth_console(2);
  fea code;
  code.run();

  return 0;
}
