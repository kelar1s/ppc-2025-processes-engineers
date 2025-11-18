#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"

namespace tabalaev_a_elem_mat_min {

TabalaevAElemMatMinMPI::TabalaevAElemMatMinMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TabalaevAElemMatMinMPI::ValidationImpl() {
  auto &rows = std::get<0>(GetInput());
  auto &columns = std::get<1>(GetInput());

  if (rows <= 0 || columns <= 0) {
    return false;
  }

  auto &matrix = std::get<2>(GetInput());

  return (rows * columns == matrix.size()) && (GetOutput() == 0);
}

bool TabalaevAElemMatMinMPI::PreProcessingImpl() {
  return true;
}

bool TabalaevAElemMatMinMPI::RunImpl() {
  auto &input = GetInput();
  auto &rows = std::get<0>(input);
  auto &columns = std::get<1>(input);
  auto &matrix = std::get<2>(input);

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t part_size = rows / world_size;
  size_t remainder = rows % world_size;

  size_t start = (world_rank * part_size) + std::min(static_cast<size_t>(world_rank), remainder);
  size_t finish = start + part_size;
  if (std::cmp_less(world_rank, remainder)) {
    finish += 1;
  }

  int local_minik = INT_MAX;

  for (size_t i = start; i < finish && i < rows; ++i) {
    for (size_t j = 0; j < columns; ++j) {
      local_minik = std::min(local_minik, matrix[(i * columns) + j]);
    }
  }

  int global_minik = 0;
  MPI_Allreduce(&local_minik, &global_minik, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_minik;

  return true;
}

bool TabalaevAElemMatMinMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_elem_mat_min
