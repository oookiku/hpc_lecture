#pragma once
#include <stdint.h>
#include "../util/util.h"
#include "block_task.h"
#include "grid_raster.h"
#include "k_split_control.h"
#include "epilogue_function.h"

namespace cutlass {
namespace gemm {

  __global__ void kernel(
                       int m,                      ///< Height in rows of op(A) and C
                       int n,                      ///< Width in columns of op(B) and C
                       int k,                      ///< Width in columns of op(A) and height in rows of op(B)
                       k_split_control k_split,    ///< Abstraction for controlling inter-block k-splitting
                       gemm::blas_scaled_epilogue op, ///< Epilogue operation to update matrix C
                       float *d_a,               ///< Pointer to matrix A array values
                       float *d_b,               ///< Pointer to matrix B array values
                       float *d_c)               ///< Pointer to matrix C array values
{
  typedef block_task<
    16,
    16,
    4> block_task_t;

    // Declare statically-allocated shared storage
    __shared__ typename block_task_t::scratch_storage_t smem;

    // Construct and run the task
    block_task_t(
        &smem,
        d_a,
        d_b,
        d_c,
        op,
        m,
        n,
        k,
        k_split).run();
}

/******************************************************************************
 * Dispatch stub
 ******************************************************************************/

/**
 * GEMM dispatch stub
 *
 * This function also serves as the autotuning entrypoint to evaluate different
 * tuning parameterizations of kernel.
 */
void dispatch(
    int             m,                              ///< Height in rows of op(A) and C
    int             n,                              ///< Width in columns of op(B) and C
    int             k,                              ///< Width in columns of op(A) and height in rows of op(B)
    float           alpha,
    float           beta,
    float         *d_a,                           ///< Device pointer to matrix A array values
    float         *d_b,                           ///< Device pointer to matrix B array values
    float         *d_c)                           ///< Device pointer to matrix C array values                      
{
    // Thread block rasterization type
    gemm::blas_scaled_epilogue epilogue(alpha, beta);
    typedef grid_raster<
        64,
        64>
        grid_raster_t;
    int max_sm_occupancy = 8;
    int sm_count;
    get_sm_count(sm_count);
    int *d_flags;
    cudaGetSymbolAddress((void**) &d_flags, d_flags_split_k);

    dim3 grid  = grid_raster_t::grid_dims(m, n);
    dim3 block = dim3(64);
    k_split_control k_split(
                            d_flags,
                            sm_count,
                            max_sm_occupancy,
                            k,
                            8,
                            block,
                            grid);
    gemm::kernel
      <<< grid,
      block>>>(
                 m,
                 n,
                 k,
                 k_split,
                 epilogue,
                 d_a,
                 d_b,
                 d_c);
}


} // namespace gemm
} // namespace cutlass
