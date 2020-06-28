#pragma once

namespace NS {

//
// global variables on device
//
extern __constant__ double _rho;
extern __constant__ double _nu;
extern __constant__ double _dx;
extern __constant__ double _dy;
extern __constant__ double _dt;
extern __constant__ int _nx;
extern __constant__ int _ny;

//
// initializer
//
void initParams(
  const double rho,
  const double nu,
  const double dx,
  const double dy,
  const double dt,
  const int nx,
  const int ny);

//
// helper functions
// 
__device__ inline bool isHalo (const int i, const int j)
{
  return (i < 1 || i >= _nx-1 || j < 1 || j >= _ny-1);
}

__device__ inline bool isOutside (const int i, const int j)
{
  return (i < 0 || i >= _nx || j < 0 || j >= _ny);
}

__device__ inline double sq(const double x)
{
  return x*x;
}

//
// solver
//
namespace Solver {

//
// calculation of velocity
//
void calcVel(
  double *un,
  double *vn,
  const double *u,
  const double *v,
  const double *p,
  const dim3 &grid,
  const dim3 &block);

__global__ void calcVelKernel(
  double *un,
  double *vn,
  const double *u,
  const double *v,
  const double *p);

//
// calculation of source term: b
// 
void calcSource(
  double *b,
  const double *u,
  const double *v,
  const dim3 &grid,
  const dim3 &block);

__global__ void calcSourceKernel(
  double *b,
  const double *u,
  const double *v);

//
// calculation of pressure
//
void calcPress(
  double *pn,
  const double *p,
  const double *b,
  const dim3 &grid,
  const dim3 &block);

__global__ void calcPressKernel(
  double *pn,
  const double *p,
  const double *b);

} // namespace Solver


//
// boundary conditons
//
namespace BC {

void cavityFlow(
  double *u,
  double *v,
  dim3 &grid,
  dim3 &block);

__global__ void cavityFlowKernel(double *u, double *v); 

void neumann(
  double *p, 
  dim3 &grid,
  dim3 &block);

__global__ void neumannKernel(double *p);

} // namespace BC

} // namespace NS
