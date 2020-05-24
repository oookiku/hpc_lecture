#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void initBucket(int *bucket)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; // i = threadIdx.x in this code
  bucket[i] = 0;
}

__global__ void setBucket(int *bucket, int *key)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x; 
  atomicAdd(&(bucket[key[i]]), 1);
}

__global__ void scan(int *ofst, int *bucket, int n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  extern __shared__ int tmp[];
  ofst[i] = bucket[i];
  __syncthreads();
  for (int j=1; j<n; j<<=1) {
    tmp[i] = ofst[i];
    __syncthreads();
    if(i >= j) ofst[i] += tmp[i-j];
    __syncthreads();
  }
  ofst[i] -= bucket[i];
}

__global__ void setKey(int *key, int *ofst, int *bucket)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  for (int j=0; j<bucket[i]; j++) {
    key[ofst[i]+j] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> h_key(n);
  int *d_key; cudaMalloc((void**) &d_key, sizeof(int)*n);
  for (int i=0; i<n; i++) {
    h_key[i] = rand() % range;
    printf("%d ",h_key[i]);
  }
  printf("\n");
  cudaMemcpy(d_key, &h_key[0], sizeof(int)*n, cudaMemcpyHostToDevice);
  int *d_bucket; cudaMalloc((void**) &d_bucket, sizeof(int)*range);
  // Initialize buckets
  initBucket<<<1, range>>>(d_bucket);
  cudaDeviceSynchronize();
  // Count the occurences of each number [0:4]
  setBucket<<<1, n>>>(d_bucket, d_key);
  cudaDeviceSynchronize();
  // Calculate offsets
  int *d_ofst; cudaMalloc((void**) &d_ofst, sizeof(int)*range);
  scan<<<1, range, range>>>(d_ofst, d_bucket, range);
  cudaDeviceSynchronize();
  // Sort the input key
  setKey<<<1, range>>>(d_key, d_ofst, d_bucket);
  cudaDeviceSynchronize();
  cudaMemcpy(&h_key[0], d_key, sizeof(int)*n, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    printf("%d ",h_key[i]);
  }
  printf("\n");
  cudaFree(d_key);
  cudaFree(d_bucket);
  cudaFree(d_ofst);
}
