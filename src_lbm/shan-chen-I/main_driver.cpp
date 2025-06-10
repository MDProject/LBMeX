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
#include "AMReX_FileIO.H"

// default grid parameters
IntVect domain_size(32);
IntVect max_box_size(16);

// default time stepping parameters
int nsteps = 5;

inline void WriteOutput(int step,
      const Geometry& geom,
			const MultiFab& hydrovs,
      StructFact& structFact, int plot_SF=0) {
  // set up variable names for output
  const int zero_avg = 1;
  const Vector<std::string> var_names = hydrovars_names(hydrovs.nComp());
  const std::string& pltfile = amrex::Concatenate("./data_mixture/plt",step,5);
  WriteSingleLevelPlotfile(pltfile, hydrovs, var_names, geom, Real(step), step);
  if (plot_SF > 0) structFact.WritePlotFile(step, static_cast<Real>(step), "plt_SF", zero_avg);
}

int main(int argc, char* argv[]) {

  amrex::Initialize(argc, argv);
  // store the current time so we can later compute total run time.
  Real strt_time = ParallelDescriptor::second();

  // read input parameters
  //ReadInput();

  // [true] for a: kBT=0; b: kBT>0 && switching on noise for the FIRST time;
  // [false] only if hope to continue from chkpoint in which noise>0;
  bool continueFromNonFluct = true;//false; 

  // Total number of steps to run; MUST be integer multiples of [plot_int];
  int nsteps = 1000000;//100000;
  int plot_int = 2000;//10;//1000; // output configurations every [plot_int] steps;
  int print_int = 100;      // print out info every [print_int] steps;
  /*specifying time window for calculating the equilibrium state solution;
    usually be set as multiples of [plot_int], from step [last_step_index-t_window] to [last_step_index];
    i.e., [numOfFrames-1]*[plot_int], in which last_step_index is also multiples of [plot_int] */
  const int t_window = 10*plot_int; 
  int out_noise_step = plot_int;    // output noise terms every [out_noise_step] steps;

  // [plot_SF_window] is the time window for calculating the structure factor;
  int plot_SF_window = 200000; // not affected by [plot_int]; out freq controlled by [out_SF_step]
  int out_SF_step = 100;

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
  MultiFab refstate(ba, dm, 2, nghost);
  MultiFab noise(ba, dm, 2*nvel, nghost);

  // set up StructFact
  int nStructVars = 11;
  const Vector<std::string> var_names = hydrovars_names(nStructVars, true);
  const Vector<int> pairA = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  const Vector<int> pairB = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  const Vector<Real> var_scaling = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0 }; // scaling factors for each variable
  StructFact structFact(ba, dm, var_names, var_scaling, pairA, pairB);

  // INITIALIZE
  LBM_init(geom, fold, gold, hydrovs);
  Print() << "\tLB step " << 0 << std::endl;
  //PrintMultiFabComp(fold, 0);
  if (plot_int > 0) WriteOutput(0, geom, hydrovs, structFact);
  //Print() << "LB initialized lattice " << domain <<"\n" << ba << dm << std::endl;

  //unit_tests(geom, hydrovs);

  // TODO: for nonhomogeneous systems perform equilibration before copying reference state

  // copy the reference state
  ParallelCopy(refstate, hydrovs, 0, 0, 2, IntVect(nghost), IntVect(nghost));
  
  // TIMESTEP
  int SF_start = nsteps - plot_SF_window; int plot_SF = 1;
  for (int step=1; step <= nsteps; ++step) {
    //Print() << "\tLB step " << step << std::endl;
    LBM_timestep(geom, fold, gold, fnew, gnew, hydrovs, refstate);
    if (step>=SF_start && step%out_SF_step == 0 && plot_SF > 0) structFact.FortStructure(hydrovs, 0);
    //PrintMultiFabComp(fold, 0);
    if (plot_int > 0 && step%plot_int ==0 && step<nsteps) {
      Print() << "\t**************************************\t" << std::endl;
      Print() << "\tLB step " << step << std::endl;
      Print() << "\t**************************************\t" << std::endl;
      WriteOutput(step, geom, hydrovs, structFact, 0);
    }
    if(step == nsteps){
      WriteOutput(step, geom, hydrovs, structFact, plot_SF);
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
