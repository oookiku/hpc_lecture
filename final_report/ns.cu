#include "ns.hpp"

namespace NS {

__constant__ double _rho;
__constant__ double _nu;
__constant__ double _dx;
__constant__ double _dy;
__constant__ double _dt;
__constant__ int _nx;
__constant__ int _ny;

void initParams(
  const double rho,
  const double nu,
  const double dx,
  const double dy,
  const double dt,
  const int nx,
  const int ny
)
{
  cudaMemcpyToSymbol(_rho,  &rho,  sizeof(double));
  cudaMemcpyToSymbol(_nu,   &nu,   sizeof(double));
  cudaMemcpyToSymbol(_dx,   &dx,   sizeof(double));
  cudaMemcpyToSymbol(_dy,   &dy,   sizeof(double));
  cudaMemcpyToSymbol(_dt,   &dt,   sizeof(double));
  cudaMemcpyToSymbol(_nx,   &nx,   sizeof(int));
  cudaMemcpyToSymbol(_ny,   &ny,   sizeof(int));
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
  const dim3 &block
)
{
  calcVelKernel<<<grid, block>>>(un, vn, u, v, p);
  cudaDeviceSynchronize();
}

__global__ void calcVelKernel(
  double *un,
  double *vn,
  const double *u,
  const double *v,
  const double *p
)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x; 
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (isHalo(i, j)) return;

  const int idx = i + _nx*j;

  const double u_c  = u[idx];
  const double u_xp = u[idx+1];
  const double u_xm = u[idx-1];
  const double u_yp = u[idx+_nx];
  const double u_ym = u[idx-_nx];
  const double v_c  = v[idx];
  const double v_xp = v[idx+1];
  const double v_xm = v[idx-1];
  const double v_yp = v[idx+_nx];
  const double v_ym = v[idx-_nx];

  // advection(convection) term
  const double adv_x = - u_c*_dt/_dx*(u_c-u_xm) - v_c*_dt/_dy*(u_c-u_ym);
  const double adv_y = - u_c*_dt/_dx*(v_c-v_xm) - v_c*_dt/_dy*(v_c-v_ym);

  // pressure term
  const double p_x = - _dt/(2.0*_rho*_dx)*(p[idx+1]-p[idx-1]);
  const double p_y = - _dt/(2.0*_rho*_dy)*(p[idx+_nx]-p[idx-_nx]);

  // diffusion term
  const double diff_x = _nu*(_dt/sq(_dx)*(u_xp-2*u_c+u_xm) + 
                           _dt/sq(_dy)*(u_yp-2*u_c+u_ym));
  const double diff_y = _nu*(_dt/sq(_dx)*(v_xp-2*v_c+v_xm) +
                           _dt/sq(_dy)*(v_yp-2*v_c+v_ym));

  un[idx] = u[idx] + adv_x + p_x + diff_x;
  vn[idx] = v[idx] + adv_y + p_y + diff_y;
}

//
// calculation of source term: b
// 
void calcSource(
  double *b,
  const double *u,
  const double *v,
  const dim3 &grid,
  const dim3 &block
)
{
  calcSourceKernel<<<grid, block>>>(b, u, v);
  cudaDeviceSynchronize();
}

__global__ void calcSourceKernel(
  double *b,
  const double *u,
  const double *v
)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x; 
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (isHalo(i, j)) return;

  const int idx = i + _nx*j;

  b[idx] = _rho*(1/_dt*((u[idx+1]-u[idx-1])/(2.0*_dx) + (v[idx+_nx]-v[idx-_nx])/(2.0*_dy))
                 - sq((u[idx+1]-u[idx-1])/(2.0*_dx)) 
                 - 2.0*(u[idx+_nx]-u[idx-_nx])/(2.0*_dy)*(v[idx+1]-v[idx-1])/(2.0*_dx)
                 - sq((v[idx+_nx]-v[idx-_nx])/(2.0*_dy)));
}

//
// calculation of Poisson eq. for pressure
//
void calcPress(
  double *pn,
  const double *p,
  const double *b,
  const dim3 &grid,
  const dim3 &block
)
{
  calcPressKernel<<<grid, block>>>(pn, p, b);
  cudaDeviceSynchronize();
}

__global__ void calcPressKernel(
  double *pn,
  const double *p,
  const double *b
)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x; 
  const int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (isHalo(i, j)) return;

  const int idx = i + _nx*j;

  pn[idx] = ((p[idx+1]+p[idx-1])*sq(_dy) + 
             (p[idx+_nx]+p[idx-_nx])*sq(_dx)) /
             (2.0*(sq(_dx)+sq(_dy))) -
             sq(_dx)*sq(_dy)/(2.0*(sq(_dx)+sq(_dy))) *
             b[idx];
}

} // namespace Solver


//
// boundary conditions
//
namespace BC {

void cavityFlow(
  double *u, 
  double *v, 
  dim3 &grid, 
  dim3 &block
)
{
  cavityFlowKernel<<<grid, block>>>(u, v);
  cudaDeviceSynchronize();
}

__global__ void cavityFlowKernel(double *u, double *v)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x; 
  const int j = blockIdx.y*blockDim.y + threadIdx.y;
  const int idx = i + _nx*j;

  // left
  if (i == 0) {
    u[idx] = 0.0;
    v[idx] = 0.0;
  }

  // right
  if (i == _nx-1) {
    u[idx] = 0.0;
    v[idx] = 0.0;
  }

  // bottom
  if (j == 0) {
    u[idx] = 0.0;
    v[idx] = 0.0;
  }

  // top
  if (j == _ny-1) {
    u[idx] = 1.0;
    v[idx] = 0.0;
  }
}

void neumann(
  double *p,
  dim3 &grid,
  dim3 &block
)
{
  neumannKernel<<<grid, block>>>(p);
  cudaDeviceSynchronize();
}

__global__ void neumannKernel(double *p)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x; 
  const int j = blockIdx.y*blockDim.y + threadIdx.y;
  const int idx = i + _nx*j;

  // left
  if (i == 0) {
    p[idx] = p[1+_nx*j];
  }

  // right
  if (i == _nx-1) {
    p[idx] = p[_nx-2+_nx*j];
  }

  // bottom
  if (j == 0) {
    p[idx] = p[i+_nx*1];
  }

  // top
  if (j == _ny-1) {
    p[idx] = 0.0;
  }
}

} // namespace BC

} // namespace NS
