#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>

// User-declared reduction is supported in OpenMP >= 4.0
#pragma omp declare reduction(+ : std::vector<int> : \
  std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
  initializer(omp_priv(omp_orig))

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range);
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }

// The following reduction is available
// in GCC 8.3.0 (my environment on tsubame)
#pragma omp parallel for reduction(+:bucket)
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
