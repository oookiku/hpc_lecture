#include <iostream>
#include <fstream>
#include <vector>
#include "ns.hpp"

using namespace std;

//
// intialization of variables on device
//
constexpr __device__ void initValue(int idx){}

template<typename Head, typename... Tail>
__device__ void initValue(int idx, Head head, Tail... tail)
{
  head[idx] = 0.0;
  initValue(idx, tail...);
}

template<typename... Args>
__global__ void initDvec(int nx, Args... args)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  const int j = blockIdx.y*blockDim.y + threadIdx.y;
  const int idx = i + nx*j;

  initValue(idx, args...);
}

int main()
{
  //
  // parameters
  //
  const double lx     = 2.0;
  const double ly     = 2.0;
  const int nx        = 41;
  const int ny        = 41;
  const double dt     = 1e-3;
  const int nt        = 700;
  const int nit       = 50;

  const double rho    = 1.0; 
  const double nu     = 0.1;

  const int bx = 16;
  const int by = 16;

  const double dx = lx/(nx-1);
  const double dy = ly/(ny-1);

  //
  // initial settings
  //
  NS::initParams(rho, nu, dx, dy, dt, nx, ny);

  const int asize = nx*ny;
  vector<double> h_u(asize, 0.0);
  vector<double> h_v(asize, 0.0);
  vector<double> h_p(asize, 0.0);

  vector<double> x(nx);
  vector<double> y(ny);
  for (int i = 0; i < nx; ++i) x[i] = i*dx;
  for (int j = 0; j < ny; ++j) y[j] = j*dx;

  double *d_un, *d_u, *d_vn, *d_v, *d_pn, *d_p, *d_b;
  cudaMalloc((void **) &d_un, asize*sizeof(double));
  cudaMalloc((void **) &d_u,  asize*sizeof(double));
  cudaMalloc((void **) &d_vn, asize*sizeof(double));
  cudaMalloc((void **) &d_v,  asize*sizeof(double));
  cudaMalloc((void **) &d_pn, asize*sizeof(double));
  cudaMalloc((void **) &d_p,  asize*sizeof(double));
  cudaMalloc((void **) &d_b,  asize*sizeof(double));

  dim3 grid((nx+bx-1)/bx, (ny+by-1)/by);
  dim3 block(bx, by);
  initDvec<<<grid, block>>>(nx, d_un, d_u, d_vn, d_v, d_pn, d_p, d_b);

  //
  // main loop
  //
  for (int n = 0; n < nt; ++n) {
    //
    // calc. pressure
    //
    NS::Solver::calcSource(d_b, d_u, d_v, grid, block);
    // iteration for solving poisson eq.
    for (int q = 0; q < nit; ++q) {
      NS::Solver::calcPress(d_pn, d_p, d_b, grid, block);
      NS::BC::neumann(d_pn, grid, block);
      swap(d_pn, d_p);
    }

    //
    // calc. velocity
    //
    NS::Solver::calcVel(d_un, d_vn, d_u, d_v, d_p, grid, block);
    NS::BC::cavityFlow(d_un, d_vn, grid, block);
    swap(d_un, d_u);
    swap(d_vn, d_v);
  }

  //
  // output result
  //
  cudaMemcpy(&h_u[0], d_u, asize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_v[0], d_v, asize*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_p[0], d_p, asize*sizeof(double), cudaMemcpyDeviceToHost);

  ofstream file;
  file.open("result2.dat", ios::out);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int idx = i + nx*j;
      file << x[i] << " " << y[j] << " " << h_u[idx] << " " << h_v[idx] << " " << h_p[idx] << endl;
    }
  }
  file.close();

  cudaFree(d_un);
  cudaFree(d_u);
  cudaFree(d_vn);
  cudaFree(d_v);
  cudaFree(d_pn);
  cudaFree(d_p);
  cudaFree(d_b);
}
