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
 * Tile-loading abstraction for thread blocks
 */

#include "../util/util.h"

namespace cutlass {
namespace gemm {


/******************************************************************************
 * block_loader (CongruousCopy + dp1 specialization)
 ******************************************************************************/

/**
 * \brief A three-phase data loading abstraction (prefetch, commit, and
 * advance) for iterating over ranges of block-wide matrix tiles.
 * (CongruousCopy + dp1 specialization)
 *
 * Each iteration sequence produces a KxL (height-by-width) block-wide tile of
 * value_t in shared memory.  The layout of the shared block-wide tile is
 * a row-major (L-major) tiling of singleton "dp1" dp_vector_t items, where
 * dp_vector_t == value_t.  Its dimensions are:
 *    K = BlockDpVectorsK
 *    L = BlockDpVectorsL
 *
 * The data is copied from a corresponding tile of global matrix data whose
 * layout of value_t is also L-major. This constitutes a CongruousCopy
 * between the L-major global tile and the L-major shared tile.
 *
 * NB: Because they are "dp1" singletons, the K-major orientation of
 * dp_vector_t in shared memory is irrelevant, and the L-major global and
 * shared tile layouts are perfectly congruous.  As a result, we can increase
 * the granularity of data transfer via vectorization of loads and stores
 * without any intermediate {dis|re}assembly.
 *
 * NB: Consecutive threads within a block are mapped in L-major
 * fashion across a first-set of LDG-vectors of dp_vector_t (value_t) within
 * their global tile. Successive sets of LDG-vectors are then strip-mined
 * as necessary down the K-axis.  These discontiguous LDG-vectors comprise the
 * thread's "slice" of the block-wide tile.
 */
template <
    int BlockThreads,           ///< Number of threads in each thread block (blockDim.x)
    int BlockDpVectorsK,        ///< Extent of block-wide tile in dp_vector_t along the K-axis (height)
    int BlockDpVectorsL>        ///< Extent of block-wide tile in dp_vector_t along the L-axis (width)
struct block_loader<
    BlockThreads,
    BlockDpVectorsK,
    BlockDpVectorsL,
    load_algorithm::CongruousCopy>  ///< Algorithm for loading a shared tile of KxL matrix data (CongruousCopy specialization)
{
    //-------------------------------------------------------------------------
    // Constants and types
    //-------------------------------------------------------------------------

    /// Dot-product vector type along the K-axis
    enum
    {
        /// Number of dp_vector_t in a block-wide tile
        BlockDpVectors = BlockDpVectorsK * BlockDpVectorsL,
    };

    enum
    {
        /// Number of dp_vector_t per io_vector
        LdgVectorDpVectors = io_vector::VectorItems,

        /// Number of value_t per io_vector
        LdgVectorItems = LdgVectorDpVectors,



        /// Total number of io_vector within each block-wide tile
        BlockLdgVectors = divide_assert<BlockDpVectors, LdgVectorDpVectors>::value,

        /// Extent of the block-wide tile in io_vector along L-axis
        BlockLdgVectorsL = divide_assert<BlockDpVectorsL, LdgVectorDpVectors>::value,

        /// Extent of the block-wide tile in io_vector along K-axis
        BlockLdgVectorsK = BlockDpVectorsK,



        /// Number of io_vector within each thread-tile
        ThreadLdgVectors = divide_assert<BlockLdgVectors, BlockThreads>::value,

        /// Extent of the thread tile in io_vector along L-axis
        ThreadLdgVectorsL = __NV_STD_MAX(1, (BlockLdgVectorsL / BlockThreads)),

        /// Extent of the thread tile in io_vector along K-axis
        ThreadLdgVectorsK = divide_assert<ThreadLdgVectors, ThreadLdgVectorsL>::value,



        /// Number of io_vector within each stripmine-tile
        StripmineLdgVectors = BlockThreads,

        /// Extent of the stripmine tile in io_vector along L-axis
        StripmineLdgVectorsL = __NV_STD_MIN(BlockLdgVectorsL, StripmineLdgVectors),

        /// Extent of the stripmine tile in io_vector along K-axis
        StripmineLdgVectorsK = divide_assert<StripmineLdgVectors, StripmineLdgVectorsL>::value,



        /// Alignment in dp_vector_t along L needed for committing prefetch
        AlignmentDpVectorsL = LdgVectorDpVectors,
    };

    /// Predicate bit vector
    typedef uint64_t predicate_mask_t;

    //-------------------------------------------------------------------------
    // Members
    //-------------------------------------------------------------------------

    /// Input pointer to matrix in io_vector
    io_vector *d_matrix_ldgvecs;

    /// Extent of the input matrix in io_vector along the L-axis
    int matrix_ldgvecs_l;

    /// Predicate bits for guarding io_vector loads within "whole-k" block-wide tiles
    predicate_mask_t guard;

    /// Predicate bits for guarding io_vector loads within the final block-wide "residue" tile
    predicate_mask_t residue_guard;

    /// Iteration span in "whole-k" block-wide tiles
    int wholek_tiles_remaining;

    /// Distance in io_vector within pitched-linear memory between successive coordinates along the K-axis
    int matrix_ldgvec_stride_k;

    /// Distance in io_vector within pitched-linear memory between successive coordinates along the L-axis
    int matrix_ldgvec_stride_l;

    /// io_vector coordinates (l, k) of thread-tile within the block-wide tile
    int2 block_thread_ldgvec_coords;

    /// Thread-wide tile of prefetch data
    io_vector thread_tile[ThreadLdgVectorsK][ThreadLdgVectorsL];


    //-------------------------------------------------------------------------
    // Constructor API
    //-------------------------------------------------------------------------

    /// Constructor
    inline __device__
    block_loader(
        float *d_matrix_items,        ///< Input pointer to matrix in value_t
        int matrix_items_l,             ///< Extent of the input matrix in value_t along the L-axis
        int matrix_items_stride_k,      ///< Distance in value_t within pitched-linear memory between successive coordinates along the K-axis
        int matrix_items_stride_l,      ///< Distance in value_t within pitched-linear memory between successive coordinates along the L-axis
        int2 matrix_block_item_coords,  ///< value_t coordinates (l, k) of first block-wide tile within the input matrix
        int block_end_item_k)           ///< Thread block's ending coordinate (k) within the input matrix (one-past)
    :
        guard(0),
        residue_guard(0)
    {
        matrix_ldgvecs_l = matrix_items_l / LdgVectorItems;
        matrix_ldgvec_stride_k = matrix_items_stride_k / LdgVectorItems,
        matrix_ldgvec_stride_l = matrix_items_stride_l;

        // io_vector coordinates (l, k) of thread-tile within the block-wide tile
        block_thread_ldgvec_coords = make_int2(
            threadIdx.x % BlockLdgVectorsL,                 // l-coordinate
            threadIdx.x / BlockLdgVectorsL);                // k-coordinate

        // io_vector coordinates (l, k) of first block-wide tile within the input matrix
        int2 matrix_block_ldgvec_coords = make_int2(
            matrix_block_item_coords.x / LdgVectorItems,     // l-coordinate
            matrix_block_item_coords.y);                    // k-coordinate

        // Iteration span in io_vector
        int span_ldgvec_k = (block_end_item_k - matrix_block_item_coords.y);



        // io_vector coordinates (l, k) of first thread-tile tile within the input matrix
        int2 matrix_thread_ldgvec_coords = make_int2(
            block_thread_ldgvec_coords.x + matrix_block_ldgvec_coords.x,
            block_thread_ldgvec_coords.y + matrix_block_ldgvec_coords.y);

        // Iteration range in "whole-k" block-wide tiles
        wholek_tiles_remaining = span_ldgvec_k / BlockLdgVectorsK;

        // Update the input pointer to be matrix_thread_ldgvec_coords
        this->d_matrix_ldgvecs =
            reinterpret_cast<io_vector*>(d_matrix_items) +
            (matrix_thread_ldgvec_coords.y * matrix_ldgvec_stride_k) +
            (matrix_thread_ldgvec_coords.x * matrix_ldgvec_stride_l);
    }


    //-------------------------------------------------------------------------
    // Loader API
    //-------------------------------------------------------------------------

    /**
     * Request the current block-wide tile
     */
    inline __device__
    void request()
    {
        // Outer thread-tile io_vector iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < ThreadLdgVectorsK; ++thread_ldgvec_k)
        {
            // Inner thread-tile io_vector iteration (L-axis)
            #pragma unroll
            for (int thread_ldgvec_l = 0; thread_ldgvec_l < ThreadLdgVectorsL; ++thread_ldgvec_l)
            {
                // Linear index of io_vector load
                int ldgvec_idx = (thread_ldgvec_k * ThreadLdgVectorsL) + thread_ldgvec_l;

                // Unpack predicate guard
                predicate_mask_t valid = ((guard >> ldgvec_idx) & 1);

                if (true || valid)
                {
                    // Perform load
                    thread_tile[thread_ldgvec_k][thread_ldgvec_l].load(
                        d_matrix_ldgvecs +
                        (thread_ldgvec_k * StripmineLdgVectorsK * matrix_ldgvec_stride_k) +
                        (thread_ldgvec_l * StripmineLdgVectorsL * matrix_ldgvec_stride_l));
                }
            }
        }
    }


    /**
     * Advance the loader to the next block-wide tile in the K-axis
     */
    inline __device__
    void next()
    {
        d_matrix_ldgvecs += (matrix_ldgvec_stride_k * BlockLdgVectorsK);
    }


    /**
     * Commit the previously-requested block-wide tile to shared memory
     *
     * NB: To facilitate padding for avoiding shared memory bank conflicts, we
     * allow the row stride SmemDpVectorsL to be arbitrarily bigger than the
     * tile width BlockDpVectorsL.
     */
    template <int SmemDpVectorsL>
    inline __device__
    void commit(
        float (&scratch_tile)[BlockDpVectorsK][SmemDpVectorsL])
    {
        // Outer thread-tile io_vector iteration (K-axis)
        #pragma unroll
        for (int thread_ldgvec_k = 0; thread_ldgvec_k < ThreadLdgVectorsK; ++thread_ldgvec_k)
        {
            int block_ldgvec_k = block_thread_ldgvec_coords.y + (thread_ldgvec_k * StripmineLdgVectorsK);

            // Inner thread-tile io_vector iteration (L-axis)
            #pragma unroll
            for (int thread_ldgvec_l = 0; thread_ldgvec_l < ThreadLdgVectorsL; ++thread_ldgvec_l)
            {
                int block_ldgvec_l = block_thread_ldgvec_coords.x + (thread_ldgvec_l * StripmineLdgVectorsL);

                thread_tile[thread_ldgvec_k][thread_ldgvec_l].store(
                    &scratch_tile[block_ldgvec_k][block_ldgvec_l * LdgVectorDpVectors]);
            }
        }
    }
};


} // namespace gemm
} // namespace cutlass
