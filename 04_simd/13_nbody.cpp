#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

namespace my {
inline float reduce_sum_f32v(__m256 v)
{
  // 8 elements -> 4 elements
  __m128 v_low1  = _mm256_castps256_ps128(v);
  __m128 v_high1 = _mm256_extractf128_ps(v, 1);
  __m128 v_sum1  = _mm_add_ps(v_low1, v_high1);
  // 4 elements -> 2 elements
  __m128 v_low2  = v_sum1;
  __m128 v_high2 = _mm_movehl_ps(v_sum1, v_sum1);
  __m128 v_sum2  = _mm_add_ps(v_low2, v_high2);
  // 2 elements -> 1 element
  __m128 v_low3  = v_sum2;
  __m128 v_high3 = _mm_shuffle_ps(v_sum2, v_sum2, 0x1);
  return _mm_cvtss_f32(_mm_add_ps(v_low3, v_high3));
}
} // namespace my

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 x_vec  = _mm256_load_ps(x);
  __m256 y_vec  = _mm256_load_ps(y);
  __m256 m_vec  = _mm256_load_ps(m);
  __m256 idx = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
  for(int i=0; i<N; i++) {
    // #0: calc 1/r
    __m256 xi_vec = _mm256_set1_ps(x[i]);
    __m256 yi_vec = _mm256_set1_ps(y[i]);
    __m256 rx_vec = _mm256_sub_ps(xi_vec, x_vec);
    __m256 ry_vec = _mm256_sub_ps(yi_vec, y_vec);
    __m256 sq_rx_vec = _mm256_mul_ps(rx_vec, rx_vec);
    __m256 sq_ry_vec = _mm256_mul_ps(ry_vec, ry_vec);
    __m256 sq_r_vec = _mm256_add_ps(sq_rx_vec, sq_ry_vec);
    __m256 rr_vec = _mm256_rsqrt_ps(sq_r_vec);
    __m256 neg = _mm256_set1_ps(-1.0);
    // #1: calc fx
    __m256 fx_tmp  = _mm256_mul_ps(rx_vec, m_vec);
    fx_tmp = _mm256_mul_ps(fx_tmp, rr_vec);
    fx_tmp = _mm256_mul_ps(fx_tmp, rr_vec);
    fx_tmp = _mm256_mul_ps(fx_tmp, rr_vec);
    fx_tmp = _mm256_mul_ps(fx_tmp, neg);
    // #2: calc fy
    __m256 fy_tmp  = _mm256_mul_ps(ry_vec, m_vec);
    fy_tmp = _mm256_mul_ps(fy_tmp, rr_vec);
    fy_tmp = _mm256_mul_ps(fy_tmp, rr_vec);
    fy_tmp = _mm256_mul_ps(fy_tmp, rr_vec);
    fy_tmp = _mm256_mul_ps(fy_tmp, neg);
    // #3: mask fx_vector and fy_vector
    __m256 i_vec = _mm256_set1_ps((float)i);
    // idx == i_vec -> 0x00000000, idx != i_vec -> 0xffffffff
    __m256 mask = _mm256_cmp_ps(idx, i_vec, _CMP_NEQ_OQ);
    fx_tmp = _mm256_and_ps(fx_tmp, mask);
    fy_tmp = _mm256_and_ps(fy_tmp, mask);
    // #4: reduce fx and fy
    fx[i] = my::reduce_sum_f32v(fx_tmp);
    fy[i] = my::reduce_sum_f32v(fy_tmp);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
