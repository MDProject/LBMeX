#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFParallelFor.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <StructFact.H>

using namespace amrex;

#include "LBM_shan-chen.H"
#include "LBM_tests.H"

// default grid parameters
IntVect domain_size(16);
IntVect max_box_size(8);

// default time stepping parameters
int nsteps = 8;//400000;

// default output parameters
int plot_SF = 50;
int plot_SF_window = 100000;
int plot_SF_start = nsteps - plot_SF_window;

int plot_int = 1;//1000;


inline void WriteOutput(int step,
      const Geometry& geom,
			const MultiFab& hydrovs,
      StructFact& structFact) {
  // set up variable names for output
  const int zero_avg = 1;
  const Vector<std::string> var_names = hydrovars_names(hydrovs.nComp());
  const std::string& pltfile = amrex::Concatenate("./data_mixture/plt",step,5);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
  if (plot_SF > 0 && step == nsteps) structFact.WritePlotFile(step, static_cast<Real>(step), "./data_mixture/plt_SF", zero_avg);
}

int main(int argc, char* argv[]) {

  amrex::Initialize(argc, argv);
  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();

  // read input parameters
  // ReadInput();

  // set up Box and Geomtry
  RealBox real_box({0.,0.,0.},{1.,1.,1.});
  IntVect dom_lo(0, 0, 0);
  IntVect dom_hi(domain_size-1);
  Array<int,3> periodicity({1,1,1});
  int nghost = 2; // need two halo layers for gradients

  Box domain(dom_lo, dom_hi);
  Geometry geom(domain, real_box, CoordSys::cartesian, periodicity);
  BoxArray ba(domain);
  ba.maxSize(max_box_size); // chop domain into boxes
  DistributionMapping dm(ba);

  // set up MultiFabs
  MultiFab fold(ba, dm, nvel, nghost);
  MultiFab fnew(ba, dm, nvel, nghost);
  MultiFab gold(ba, dm, nvel, nghost);
  MultiFab gnew(ba, dm, nvel, nghost);
  MultiFab hydrovs(ba, dm, 2*nvel, nghost);
  MultiFab noise(ba, dm, 2*nvel, nghost);

  // set up StructFact
  int nStructVars = 5;
  const Vector<std::string> var_names = hydrovars_names(20);
  for (const auto& name : var_names) {
    std::cout << name << std::endl;
  }
  const Vector<int> pairA = { 0, 1, 2, 3, 4 };
  const Vector<int> pairB = { 0, 1, 2, 3, 4 };
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0 };
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);

  // INITIALIZE
  LBM_init(geom, fold, gold, hydrovs);
  if (::plot_int > 0) WriteOutput(0, geom, hydrovs, structFact);
  Print() << "LB initialized lattice " << domain <<"\n" << ba << dm << std::endl;

  unit_tests(geom, hydrovs);

  // TIMESTEP
  for (int step=1; step <= nsteps; ++step) {
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs);
    if (plot_SF > 0 && step >= plot_SF_start && step%plot_SF == 0) structFact.FortStructure(hydrovs, 0);
    // not the hydrovs, but real hydrovars??? 
    if (::plot_int > 0 && step%(::plot_int) == 0) {
      WriteOutput(step, geom, hydrovs, structFact);
      Print() << "LB step " << step << std::endl;
    }
  }

  Print() << "LB completed " << nsteps << " time steps" << std::endl;

  // Call the timer again and compute the maximum difference between the start time 
  // and stop time over all processors
  Real stop_time = ParallelDescriptor::second() - strt_time;
  ParallelDescriptor::ReduceRealMax(stop_time);
  amrex::Print() << "Run time = " << stop_time << " s (" << domain.numPts()*nsteps/stop_time << " LUP/s)" << std::endl;
  
  amrex::Finalize();
}
