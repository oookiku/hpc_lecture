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
 * \brief I/O device intrinsics
 */

#include <stdint.h>
#include <cuda_fp16.h>

#include "nv_std.h"
#include "math.h"

namespace cutlass {




/******************************************************************************
 * io_vector
 ******************************************************************************/

/**
 * \brief Aligned vector type for coarsening data movement instructions
 *
 * Exposes the member constant \p VectorItems, the actual number of component
 * values comprising the io_vector
 */
struct io_vector
{
    enum
    {
        VectorItems = 4,
        AlignBytes = 16
    };

    float __align__(16) buff[VectorItems];

    inline __device__
    void load(const io_vector *ptr)
    {
        *this = *ptr;
    }

    inline __device__
    void load(const float *ptr)
    {
        *this = *reinterpret_cast<const io_vector*>(ptr);
    }

    inline __device__
    void store(float *ptr) const
    {
        *reinterpret_cast<io_vector*>(ptr) = *this;
    }

};




/******************************************************************************
 * I/O cast types
 ******************************************************************************/

/// Provides the type for which to reinterpret-cast a given vector
struct io_cast
{
    typedef float type[1];
};

/******************************************************************************
 * ldg_cg intrinsics
 ******************************************************************************/

inline __device__
void ldg_cg_internal(
    float (&dest)[1], 
    float *ptr)
{
    asm volatile ("ld.global.cg.f32 %0, [%1];\n"
        :
            "=f"(dest[0])
        :
            "l"(ptr));
}

/// Load from global (cache-global modifier)
inline __device__
void ldg_cg(
    float &dest,
    float *d_in)
{
    // Cast dest to a different array type if necessary
    ldg_cg_internal(
        reinterpret_cast<typename io_cast::type &>(dest),
        d_in);
}


/******************************************************************************
 * stg_cg intrinsics
 ******************************************************************************/

inline __device__
void stg_cg_internal(
    float *ptr,
    const float (&src)[1])
{
    asm volatile ("st.global.cg.f32 [%0], %1;\n"
        : :
            "l"(ptr),
            "f"(src[0]));
}

/// Store to global (cache-global modifier)
inline __device__
void stg_cg(
    float *dest,
    const float &src)
{
    // Cast src to a different array type if necessary
    stg_cg_internal(
        dest,
        reinterpret_cast<const typename io_cast::type &>(src));
}


} // namespace cutlass
