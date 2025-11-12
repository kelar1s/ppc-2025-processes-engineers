#include "tabalaev_a_elem_mat_min/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"

namespace tabalaev_a_elem_mat_min {

TabalaevAElemMatMinSEQ::TabalaevAElemMatMinSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TabalaevAElemMatMinSEQ::ValidationImpl() {
  auto &rows = std::get<0>(GetInput());
  auto &columns = std::get<1>(GetInput());

  if (rows <= 0 || columns <= 0) {
    return false;
  }

  auto &matrix = std::get<2>(GetInput());

  return (rows * columns == matrix.size()) && (GetOutput() == 0);
}

bool TabalaevAElemMatMinSEQ::PreProcessingImpl() {
  return true;
}

bool TabalaevAElemMatMinSEQ::RunImpl() {
  auto &matrix = std::get<2>(GetInput());

  int minik = *std::min_element(matrix.begin(), matrix.end());

  GetOutput() = minik;
  return true;
}

bool TabalaevAElemMatMinSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_elem_mat_min
