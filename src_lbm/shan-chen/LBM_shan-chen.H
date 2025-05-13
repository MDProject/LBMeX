#ifndef LBM_H_
#define LBM_H_

#include "LBM_d3q19.H"

const IntVect print_grid(0, 0, 0); // grid to print out

// default initial condition
std::string init_cond = "mixture";

AMREX_GPU_MANAGED Real G = 0.8;//1.5;
AMREX_GPU_MANAGED Real rho_hi = 1.;//2.3; // values reported by Belardinelli et al. [Phys. Rev. E 91, 023313 (2015)]
AMREX_GPU_MANAGED Real rho_lo = 1.;//0.06;
AMREX_GPU_MANAGED Real sigma = 0.174; // interfacial tension
AMREX_GPU_MANAGED Real alpha = 6; // interface width

AMREX_GPU_MANAGED Real temperature = 0.;//1e-5;

AMREX_GPU_MANAGED Real init_frac = 0.5; // fractional size of droplet or stripe

AMREX_GPU_CONSTANT Real const lambda_d = 1.0;
AMREX_GPU_CONSTANT Real const lambda_b = 1.0;
AMREX_GPU_CONSTANT Real const lambda_s = 1.0;
AMREX_GPU_CONSTANT Real const lambda_o = 1.0;
AMREX_GPU_CONSTANT Real const lambda_e = 1.0;

AMREX_GPU_CONSTANT Array<Real,nvel> const
lambda_f = { 0.,
  lambda_d, lambda_d, lambda_d,
  lambda_b, lambda_s, lambda_s, lambda_s, lambda_s, lambda_s,
  lambda_o, lambda_o, lambda_o, lambda_o, lambda_o,
  lambda_e, lambda_e, lambda_e, lambda_e
};

AMREX_GPU_CONSTANT Array<Real,nvel> const
lambda_g = { 0.,
  lambda_d, lambda_d, lambda_d,
  lambda_b, lambda_s, lambda_s, lambda_s, lambda_s, lambda_s,
  lambda_o, lambda_o, lambda_o, lambda_o, lambda_o,
  lambda_e, lambda_e, lambda_e, lambda_e
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RealVect gradient(int x, int y, int z, const Array4<Real>& field, int icomp) {
  RealVect gradient = { 0.0, 0.0, 0.0 };
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    /*if(x==-1&&y==-1&&z==-1){
      printf("density used at (%d,%d,%d) = %.8f\n", xp, yp, zp, field(xp,yp,zp,icomp));
    }*/
    for (int dir=0; dir<3; dir++) {
      gradient[dir] += w[i]/cs2*field(xp,yp,zp,icomp)*c[i][dir];
    }
  }
  return gradient;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real laplacian(int x, int y, int z, const Array4<Real>& field, int icomp) {
  Real laplacian = 0.0;
  for (int i=0; i<nvel; i++) {
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    laplacian += 2.*w[i]/cs2*(field(xp,yp,zp,icomp)-field(x,y,z,icomp));
  }
  return laplacian;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array<Array1D<Real,0,nvel>,2> force_moments(const Array<Real,2>& rhoa, Array<RealVect,2>& va, const Array<RealVect,2>& aa, const Array1D<Real, 0, 2*nvel>& xia, bool tag=false) {
  Array<Array1D<Real,0,nvel>,2> phia;
  Array<RealVect,2> va_tmp = va;  // [va] denotes the modified velocity
  for (int i=0; i<2; ++i) {
    int j = (i+1)%2;

    RealVect xi;
    for (int k=0; k<AMREX_SPACEDIM; ++k) xi[k] = xia(i*nvel+k+1);

    //printf("xi(%d) = %f\t", i, xi[i]);  // only has nonzero values for i=0 ???

    // express v, v' in terms of v̄, v̄' [Belardinelli et al. Eq. (71)] ?????????????????????????    last term????  + or - ???
    // and va[0] will be changed after first loop ?
    va_tmp[i] = va[i] + 0.5*( aa[i] - lambda_d*rhoa[j]/(rhoa[i]+rhoa[j])*(va[i]-va[j]+0.5*(aa[i]-aa[j])) + (1-0.5*lambda_d)*xi/rhoa[i] );

    Array1D<Real,0,nvel>& phi = phia[i];
    Real const& rho = rhoa[i];
    RealVect const& v = va_tmp[i];
    RealVect const& a = aa[i];

    if(tag){
      for(int n=0; n<3; ++n) {
        if(i==0){
          printf("vf(%d) = %.8f\t", n, v[n]);
        }else{
          printf("vg(%d) = %.8f\t", n, v[n]);
        }
      }
    }
    /* density mode */
    phi(0) = 0.0;

    /* momentum modes */
    phi(1) = rho*a[0];
    phi(2) = rho*a[1];
    phi(3) = rho*a[2];

    /* stress modes */  // ???? 
    Real factor = 2.;
    phi(4) = factor*rho*(a[0]*v[0] + a[1]*v[1] + a[2]*v[2]);
    phi(5) = factor*rho*(2.*a[0]*v[0] - a[1]*v[1] - a[2]*v[2]);
    phi(6) = factor*rho*(a[1]*v[1] - a[2]*v[2]);
    phi(7) = factor*0.5*rho*(a[0]*v[1]+a[1]*v[0]);
    phi(8) = factor*0.5*rho*(a[1]*v[2]+a[2]*v[1]);
    phi(9) = factor*0.5*rho*(a[2]*v[0]+a[0]*v[2]);

    /* kinetic modes  */
    for (int i=10; i<nvel; ++i) phi(i) = 0.0;

  }

  return phia;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,2*nvel> thermal_noise(const Array<Real,2>& rho, const RandomEngine& engine) {
  Array1D<Real,0,2*nvel> xi = {};

  // densities are conserved
  xi(0) = xi(nvel) = 0.0;

  // fluxes are anticorrelated (total momentum is conserved)
  for (int i=1; i<=AMREX_SPACEDIM; ++i ) {
    const Real pref = (2.-lambda_d)*lambda_d*temperature*0.5;//rho[0]*rho[1]/(rho[0]+rho[1]);
    xi(i) = std::sqrt(pref) * RandomNormal(0., 1., engine);
    xi(i+nvel) = - xi(i);
  }

  // remaining modes
  for (int i=AMREX_SPACEDIM+1; i<nvel; ++i) {
    const Real pref = (2.-lambda_f[i])*lambda_f[i]*b[i]*temperature/cs2;//*rho[0];
    const Real preg = (2.-lambda_g[i])*lambda_g[i]*b[i]*temperature/cs2;//*rho[1];
    xi(i) = std::sqrt(pref) * RandomNormal(0., 1., engine);
    xi(i+nvel) = std::sqrt(preg) * RandomNormal(0., 1., engine);
  }

  return xi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Array1D<Real,0,2*nvel> thermal_noise() {
  Array1D<Real,0,2*nvel> xi = {};

  for(int i=0; i<2*nvel; ++i) {
    xi(i) = 0.0;
  }

  return xi;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide(int x, int y, int z,
	     const Array4<Real>& f,
	     const Array4<Real>& g,
       const Array4<Real>& h/*,
       const RandomEngine& engine*/) {
  Array1D<Real,0,nvel> mf, mg, mfEq, mgEq;
  Array<Array1D<Real,0,nvel>,2> phi;
  Array1D<Real,0,2*nvel> xi;
  Array<Real,2> rho;
  Array<RealVect,2> v, a;
  RealVect v_b, a_b;

  mf = moments(x,y,z,f); // redundant ? --> h(x,y,z,0) = mf(0) = rho[0];
  mg = moments(x,y,z,g);

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]){
    printf("moments before collision = \n");
    for(int i=0; i<nvel; ++i) {
      printf("mf(%d) = %f\t", i, mf(i));
      printf("mg(%d) = %f\t", i, mg(i));
    }
    printf("\n");
  }

  rho[0] = mf(0);
  rho[1] = mg(0);

  v[0] = (mf(0) > FLT_EPSILON) ? RealVect(mf(1), mf(2), mf(3))/mf(0) : RealVect(0.);
  v[1] = (mg(0) > FLT_EPSILON) ? RealVect(mg(1), mg(2), mg(3))/mg(0) : RealVect(0.);
  //v[0] = RealVect(h(x,y,z,2), h(x,y,z,3), h(x,y,z,4));
  //v[1] = RealVect(h(x,y,z,5), h(x,y,z,6), h(x,y,z,7));

  a[0] = -G*cs2*gradient(x,y,z,h,1);
  a[1] = -G*cs2*gradient(x,y,z,h,0);
  //a[0] = RealVect(h(x,y,z,2*AMREX_SPACEDIM+2), h(x,y,z,2*AMREX_SPACEDIM+3), h(x,y,z,2*AMREX_SPACEDIM+4));
  //a[1] = RealVect(h(x,y,z,2*AMREX_SPACEDIM+5), h(x,y,z,2*AMREX_SPACEDIM+6), h(x,y,z,2*AMREX_SPACEDIM+7));

  v_b = (rho[0]*v[0]+rho[1]*v[1])/(rho[0]+rho[1]);
  a_b = (rho[0]*a[0]+rho[1]*a[1])/(rho[0]+rho[1]);

  mfEq = equilibrium_moments(rho[0], v_b+0.5*a_b);  // here v[*], a[*] are modified quantities; so additional 0.5*a_b is required;
  mgEq = equilibrium_moments(rho[1], v_b+0.5*a_b);

  xi = thermal_noise();//thermal_noise(rho, engine); // TODO: need to use the reference densities [uschill 2025/01/12]

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]){
    printf("fluid f,g velocity in collide() = \n");
    phi = force_moments(rho, v, a, xi, true);
    printf("\n");
  }else{
    phi = force_moments(rho, v, a, xi);
  }

  Array1D<Real,0,nvel> const& phif = phi[0];
  Array1D<Real,0,nvel> const& phig = phi[1];

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]) {
    printf("moments in collide() = \n");
    for(int i=0; i<nvel; ++i) {
      printf("mfEq(%d) = %.8f\t", i, mfEq(i));
      printf("phif(%d) = %.8f\t", i, phif(i));
      printf("mgEq(%d) = %.8f\t", i, mgEq(i));
      printf("phig(%d) = %.8f\t", i, phig(i));
    }
    printf("\ndensity in collide() = \n");
    printf("h(%d) = %.8f\t", 0, h(x,y,z,0));
    printf("h(%d) = %.8f\t", 1, h(x,y,z,1));
    printf("\ncenter-of-mass velocity & acceleration in collide() = \n");
    RealVect const& vr_b = v_b+0.5*a_b;
    for(int i=0; i<3; ++i) {
      printf("v_b(%d) = %.8f\t", i, vr_b[i]);
      printf("af(%d) = %.8f\t", i, a[0][i]);
      printf("ag(%d) = %.8f\t", i, a[1][i]);
    }
    printf("\n");
  }

  for (int i=0; i<nvel; ++i) {
    mf(i) = mf(i) - lambda_f[i] * (mf(i) - mfEq(i)) + (1.-0.5*lambda_f[i])*phif(i) + xi(i);
    mg(i) = mg(i) - lambda_g[i] * (mg(i) - mgEq(i)) + (1.-0.5*lambda_g[i])*phig(i) + xi(i+nvel);
  }

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]) {
    printf("moments after collision = \n");
    for(int i=0; i<nvel; ++i) {
      printf("mf(%d) = %f\t", i, mf(i));
      printf("mg(%d) = %f\t", i, mg(i));
    }
    printf("\n");
  }

  populations(x,y,z,f,mf);
  populations(x,y,z,g,mg);

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]) {
    printf("populations after collision = \n");
    for(int i=0; i<nvel; ++i) {
      printf("f(%d) = %.8f\t", i, f(x,y,z,i));
      printf("g(%d) = %.8f\t", i, g(x,y,z,i));
    }
    printf("\n");
  }

}


// stream from \vec{x} to \vec{x}+\vec{c}_i (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void stream_push(int x, int y, int z,
		 const Array4<Real>& fOld, const Array4<Real>& gOld,
		 const Array4<Real>& fNew, const Array4<Real>& gNew) {
  for (int i=0; i<nvel; ++i) { // push populations to neighbors
    int xp = x + c[i][0];
    int yp = y + c[i][1];
    int zp = z + c[i][2];
    fNew(xp,yp,zp,i) = fOld(x,y,z,i);
    gNew(xp,yp,zp,i) = gOld(x,y,z,i);
    //printf("streaming from (%d, %d, %d) to (%d, %d, %d) is %.8f for comp = %d\n", x, y, z, xp, yp, zp, fOld(x,y,z,i), i);
  }
}

// collide and stream (push scheme)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void collide_stream(int x, int y, int z,
		    const Array4<Real>& fOld,  const Array4<Real>& gOld,
		    const Array4<Real>& fNew,  const Array4<Real>& gNew,
		    const Array4<Real>& h/*,
        const RandomEngine& engine*/) {
  collide(x, y, z, fOld, gOld, h/*, engine*/); // collide populations on current site
  stream_push(x, y, z, fOld, gOld, fNew, gNew); // push populations to neighbors
}

// apply fluctuating lattice Boltzmann equation for whole lattice
inline void collide_stream(const Geometry& geom,
			   MultiFab& fold, MultiFab& gold,
			   MultiFab& fnew, MultiFab& gnew,
			   MultiFab& hydrovs) {
  const int halo = 1; // need to push from first halo layer into domain
  fold.FillBoundary(geom.periodicity());
  gold.FillBoundary(geom.periodicity());
  hydrovs.FillBoundary(geom.periodicity());
  /*for (MFIter mfi(fold); mfi.isValid(); ++mfi) {
    const Array4<Real>& fOld = fold.array(mfi);
    const Array4<Real>& gOld = gold.array(mfi);
    const Array4<Real>& fNew = fnew.array(mfi);
    const Array4<Real>& gNew = gnew.array(mfi);
    const Array4<Real>& h = hydrovs.array(mfi);
    const Box& valid_box = mfi.growntilebox(halo);
    //printf("ParallelFor loop start: = \n");
    ParallelForRNG(valid_box, [=] AMREX_GPU_DEVICE(int x, int y, int z, const RandomEngine& engine) {
      //printf("(%d, %d, %d)\t", x, y, z);
      collide_stream(x, y, z, fOld, gOld, fNew, gNew, h, engine);
    });
    //printf("ParallelFor loop end \n");
  }*/

  auto const & fOld = fold.arrays();
  auto const & gOld = gold.arrays();
  auto const & fNew = fnew.arrays();
  auto const & gNew = gnew.arrays();
  auto const & h = hydrovs.arrays();
  ParallelFor(fold, IntVect(halo), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    //printf("(%d,%d,%d)\t", x, y, z);
    collide_stream(x, y, z,
      fOld[nbx], gOld[nbx],
      fNew[nbx], gNew[nbx], h[nbx]);
  });

  MultiFab::Swap(fold, fnew, 0, 0, nvel, 0);
  MultiFab::Swap(gold, gnew, 0, 0, nvel, 0);
}

inline Vector<std::string> hydrovars_names(const int numVars) {
  Vector<std::string> var_names(numVars);
  std::string name;
  int cnt = 0;
  // rho, phi
  if (cnt<numVars) var_names[cnt++] = "rhoA";
  if (cnt<numVars) var_names[cnt++] = "rhoB";
  // u
  for (int d=0; d<AMREX_SPACEDIM && cnt<numVars; d++) {
    name = "uA";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // j
  for (int d=0; d<AMREX_SPACEDIM && cnt<numVars; d++) {
    name = "uB";
    name += (120+d);
    var_names[cnt++] = name;
  }
  // remaining moments
  for (; cnt<nvel+AMREX_SPACEDIM+1 && cnt<numVars;) {
    name = "mf" + std::to_string(cnt-(AMREX_SPACEDIM+1));
    var_names[cnt++] = name;
  }
  for (; cnt<2*nvel && cnt<numVars;) {
    name = "mg"+std::to_string(cnt-nvel);
    var_names[cnt++] = name;
  }
  return var_names;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void hydrovars(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h) {
  const Array1D<Real,0,nvel> mf = moments(x,y,z,f);
  const Array1D<Real,0,nvel> mg = moments(x,y,z,g);
  h(x,y,z,0) = mf(0);
  h(x,y,z,1) = mg(0);
  for (int i=1; i<=AMREX_SPACEDIM; ++i) {     // modified momentum/velocity i=1..3
    h(x,y,z,i+1) = mf(i)/mf(0);               // u_f to be stored in 2..4
    h(x,y,z,i+1+AMREX_SPACEDIM) = mg(i)/mg(0);// u_g to be stored in 5..7
  }

  if(x==print_grid[0] && y==print_grid[1] && z==print_grid[2]) {
    printf("populations after streaming in hydrovars() = \n");
    for(int i=0; i<nvel; ++i) {
      printf("hydro_f(%d) = %.8f\t", i, f(x,y,z,i));
      printf("hydro_g(%d) = %.8f\t", i, g(x,y,z,i));
    }
    printf("\ndensity after streaming in hydrovars() = \n");
    printf("hydro(%d) = %.8f\t", 0, h(x,y,z,0));
    printf("hydro(%d) = %.8f\t", 1, h(x,y,z,1));
    printf("\n");
  }
}

// calculate hydrodynamic moments for whole lattice
inline void hydrovars(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  ParallelFor(hydrovs, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    hydrovars(x, y, z, f[nbx], g[nbx], h[nbx]);
  });
  //hydrovs.FillBoundary(geom.periodicity());
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void LBM_hydrovars_force(int x, int y, int z,
	       const Array4<Real>& f,
	       const Array4<Real>& g,
	       const Array4<Real>& h) {
  
  Array1D<Real,0,AMREX_SPACEDIM> afbar;
  Array1D<Real,0,AMREX_SPACEDIM> agbar;

  const Array1D<RealVect,0,2> grad = {
    gradient(x,y,z,h,0),
    gradient(x,y,z,h,1),
  };
  // acceleration of fluid f;
  const RealVect grad_rho = grad(0);
  // acceleration of fluid g;
  const RealVect grad_phi = grad(1);

  for(int k=0; k<AMREX_SPACEDIM; k++){
    afbar(k) = -cs2*(G*grad_phi[k]);
    agbar(k) = -cs2*(G*grad_rho[k]);
  }

  h(x,y,z,2*AMREX_SPACEDIM+2) = afbar(0);  // a_f to be stored in 8,9,10
  h(x,y,z,2*AMREX_SPACEDIM+3) = afbar(1);
  h(x,y,z,2*AMREX_SPACEDIM+4) = afbar(2);
  h(x,y,z,2*AMREX_SPACEDIM+5) = agbar(0);  // a_g to be stored in 11,12,13
  h(x,y,z,2*AMREX_SPACEDIM+6) = agbar(1);
  h(x,y,z,2*AMREX_SPACEDIM+7) = agbar(2);
  
}

// calculate modified hydrodynamic moments for whole lattice
inline void LBM_hydrovars_force(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovs) {
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  auto const & h = hydrovs.arrays();
  const IntVect halo = IntVect(2);

  //  First step: update the density first (due to the fact that the accerleration field depends on a cluster of grids rather than single grid)
  ParallelFor(hydrovs, halo, [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) { 
    LBM_hydrovars_force(x, y, z, f[nbx], g[nbx], h[nbx]);
  });
  hydrovs.FillBoundary(geom.periodicity());
}

// LB timestep for whole lattice
inline void LBM_timestep(const Geometry& geom,
			 MultiFab& fold, MultiFab& gold,
			 MultiFab& fnew, MultiFab& gnew,
			 MultiFab& hydrovs) {
  collide_stream(geom, fold, gold, fnew, gnew, hydrovs);
  hydrovars(geom, fold, gold, hydrovs);
  //LBM_hydrovars_force(geom, fold, gold, hydrovs);
}

// homogeneous mixture of C1 and C2 with random concentration fluctuations
inline void LBM_init_mixture(const Geometry& geom,
           MultiFab& mf,  MultiFab& mg,
			     MultiFab& hydrovs) {

  printf("LBM_init_mixture() called\n");
  const Real rho = 0.5*(::rho_hi+::rho_lo); // =1
  for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const Array4<Real>& f = mf.array(mfi);
    const Array4<Real>& g = mg.array(mfi);
    ParallelForRNG(mfi.validbox(), [=] AMREX_GPU_DEVICE(int x, int y, int z, RandomEngine const& engine) {
      const Real noise = 0.;//0.01*RandomNormal(0., 1., engine);
      for (int i=0; i<nvel; i++) {
        f(x,y,z,i) = w[i]*(rho + (x+y+z)/500.);
        g(x,y,z,i) = w[i]*(rho - (x+y+z)/500.);
      }
    });
  }
  hydrovars(geom, mf, mg, hydrovs);
  LBM_hydrovars_force(geom, mf, mg, hydrovs);
}

// flat interface between C1 and C2
inline void LBM_init_stripe(const Real frac,
            const Geometry& geom,
				    MultiFab& mf,
				    MultiFab& mg,
				    MultiFab& hydrovs) {
  GpuArray<Real,3> dx = geom.CellSizeArray();
  const IntVect box = geom.Domain().length();
  const Real rho_t = ::rho_hi + ::rho_lo;
  const Real pos_lo = -0.5*frac;
  const Real pos_hi = 0.5*frac;
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real pos = z-box[2]/2;
    const Real rho =  (::rho_hi-::rho_lo)*0.5*(std::tanh((pos-pos_lo)/std::sqrt(alpha)) + std::tanh((pos_hi-pos)/std::sqrt(alpha))) + ::rho_lo;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*(rho_t-rho);
    }
  });
  hydrovars(geom, mf, mg, hydrovs);
}

// binary droplet of C1 and C2
inline void LBM_init_droplet(const Real dfrac,
			     const Geometry& geom,
			     MultiFab& mf,
			     MultiFab& mg,
			     MultiFab& hydrovs) {
  GpuArray<Real,3> dx = geom.CellSizeArray();
  const IntVect box = geom.Domain().length();
  const Real R = 0.5*dfrac*box[0];
  const Real rho_t = ::rho_hi + ::rho_lo;
  auto const & f = mf.arrays();
  auto const & g = mg.arrays();
  ParallelFor(mf, IntVect(0), [=] AMREX_GPU_DEVICE(int nbx, int x, int y, int z) {
    const Real rx = x-box[0]/2;
    const Real ry = y-box[1]/2;
    const Real rz = z-box[2]/2;
    const Real r = std::sqrt(rx*rx+ry*ry+rz*rz);
    const Real rho = (::rho_hi-::rho_lo)*0.5*(1.+std::tanh((R-r)/std::sqrt(alpha))) + ::rho_lo;
    for (int i=0; i<nvel; i++) {
      f[nbx](x,y,z,i) = w[i]*rho;
      g[nbx](x,y,z,i) = w[i]*(rho_t-rho);
    }
  });
  hydrovars(geom, mf, mg, hydrovs);
}

inline void LBM_init(const Geometry& geom, MultiFab& mf, MultiFab& mg, MultiFab& hydrovs) {
  if (init_cond == "mixture") LBM_init_mixture(geom, mf, mg, hydrovs);
  else if (init_cond == "stripe") LBM_init_stripe(init_frac, geom, mf, mg, hydrovs);
  else if (init_cond == "droplet") LBM_init_droplet(init_frac, geom, mf, mg, hydrovs);
  else {
    Print() << "ERROR: Unkown init_cond";
    ParallelDescriptor::Abort(-1, false);
  }
}

#endif
