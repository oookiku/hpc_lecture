/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * Abstraction for coordinating inter-block k-splitting
 */

#include <stdint.h>

#include "../util/util.h"

namespace cutlass {
namespace gemm {


/******************************************************************************
 * Storage and initialization
 ******************************************************************************/

enum
{
    NumFlagsSplitK = 4096
};


/******************************************************************************
 * k_split_control
 ******************************************************************************/

/**
 * \brief Abstraction for coordinating inter-block k-splitting
 */
struct k_split_control
{
    /// Extent of a thread block's partition along the GEMM K-axis
    int split_k;

    /// Pointer to semaphore
    int *d_flags;



    //-------------------------------------------------------------------------
    // Device API
    //-------------------------------------------------------------------------

    /**
     * Return the thread block's starting coordinate (k) within the
     * multiplicand matrices
     */
    inline __device__
    int block_begin_item_k()
    {
        return blockIdx.z * split_k;
    }


    /**
     * Return the thread block's ending coordinate (k) within the multiplicand
     * matrices (one-past)
     */
    inline __device__
    int block_end_item_k(int dim_k)
    {
        int next_start_k = block_begin_item_k() + split_k;
        return __NV_STD_MIN(next_start_k, dim_k);
    }

    //-------------------------------------------------------------------------
    // Grid launch API
    //-------------------------------------------------------------------------

    /**
     * Constructor
     */
    inline
    k_split_control(
        int     sm_count,
        int     max_sm_occupancy,
        int     dim_k,
        int     block_tile_items_k,
        dim3    block_dims,
        dim3    &grid_dims)         ///< [in,out]
    :
        split_k(dim_k)
    {
        // Compute wave efficiency
        float wave_efficiency = get_wave_efficiency(
            sm_count,
            max_sm_occupancy,
            block_dims,
            grid_dims);

        // Update split-k if wave efficiency is less than some threshold
        if (wave_efficiency < 0.9)
        {
            int num_threadblocks = grid_dims.x * grid_dims.y * grid_dims.z;

            // Ideal number of thread blocks in grid
            int ideal_threadblocks = lcm(sm_count, num_threadblocks);

            // Desired number of partitions to split K-axis into
            int num_partitions = ideal_threadblocks / num_threadblocks;

            // Compute new k-split share
            int new_split_k = (dim_k + num_partitions - 1) / num_partitions;

            // Round split_k share to the nearest block_task_policy_t::BlockItemsK
            new_split_k = round_nearest(new_split_k, block_tile_items_k);

            // Recompute k-splitting factor with new_split_k
            num_partitions = (dim_k + new_split_k - 1) / new_split_k;

            // Update grid dims and k if we meet the minimum number of iterations worth the overhead of splitting
            int min_iterations_k = 8;

            if (((new_split_k / block_tile_items_k) > min_iterations_k) &&    // We're going to go through at least this many k iterations
                (sm_count * max_sm_occupancy < NumFlagsSplitK))             // We have enough semaphore flags allocated
            {
                grid_dims.z = num_partitions;
                split_k = new_split_k;
            }
        }
    }

    /**
     * Compute the efficiency of dispatch wave quantization
     */
    float get_wave_efficiency(
        int     sm_count,
        int     max_sm_occupancy,
        dim3    block_dims,
        dim3    grid_dims)
    {
        // Heuristic for how many warps are needed to saturate an SM for a given
        // multiply-accumulate genre.  (NB: We could make this more rigorous by
        // specializing on data types and SM width)
        int saturating_warps_per_sm = 16;

        int num_threadblocks                = grid_dims.x * grid_dims.y * grid_dims.z;
        int threads_per_threadblock         = block_dims.x * block_dims.y;
        int warps_per_threadblock           = threads_per_threadblock / 32;
        int saturating_threadblocks_per_sm  = (saturating_warps_per_sm + warps_per_threadblock - 1) / warps_per_threadblock;

        int saturating_residency    = sm_count * saturating_threadblocks_per_sm;
        int full_waves              = num_threadblocks / saturating_residency;
        int remainder_threadblocks          = num_threadblocks % saturating_residency;
        int total_waves             = (remainder_threadblocks == 0) ? full_waves : full_waves + 1;

        float last_wave_saturating_efficiency = float(remainder_threadblocks) / saturating_residency;

        return (float(full_waves) + last_wave_saturating_efficiency) / total_waves;
    }
};


} // namespace gemm
} // namespace cutlass
