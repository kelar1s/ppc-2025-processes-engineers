#include "tabalaev_a_cannon_mat_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"

namespace tabalaev_a_cannon_mat_mul {

TabalaevACannonMatMulMPI::TabalaevACannonMatMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevACannonMatMulMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int validation = 0;

  if (world_rank == 0) {
    const auto N = std::get<0>(GetInput());
    const auto &A = std::get<1>(GetInput());
    const auto &B = std::get<2>(GetInput());

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int q = static_cast<int>(std::sqrt(world_size));

    if (N > 0 && A.size() == N * N && B.size() == N * N && q * q == world_size && N % q == 0) {
      validation = 1;
    }
  }

  MPI_Bcast(&validation, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return validation != 0;
}

bool TabalaevACannonMatMulMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TabalaevACannonMatMulMPI::RunImpl() {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  return true;
}

bool TabalaevACannonMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_cannon_mat_mul
