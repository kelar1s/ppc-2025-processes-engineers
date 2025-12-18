#include "tabalaev_a_cannon_mat_mul/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"

namespace tabalaev_a_cannon_mat_mul {

TabalaevACannonMatMulSEQ::TabalaevACannonMatMulSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevACannonMatMulSEQ::ValidationImpl() {
  const auto N = std::get<0>(GetInput());
  const auto &A = std::get<1>(GetInput());
  const auto &B = std::get<2>(GetInput());

  return N > 0 && A.size() == N * N && B.size() == N * N;
}

bool TabalaevACannonMatMulSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool TabalaevACannonMatMulSEQ::RunImpl() {
  const auto N = std::get<0>(GetInput());
  const auto &A = std::get<1>(GetInput());
  const auto &B = std::get<2>(GetInput());
  std::vector<double> C(N * N, 0.0);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }

  GetOutput() = C;
  return true;
}

bool TabalaevACannonMatMulSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_cannon_mat_mul
