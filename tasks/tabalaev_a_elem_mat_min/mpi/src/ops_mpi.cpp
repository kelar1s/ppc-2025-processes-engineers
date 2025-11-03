#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_elem_mat_min {

TabalaevAElemMatMinMPI::TabalaevAElemMatMinMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TabalaevAElemMatMinMPI::ValidationImpl() {
  auto& rows = std::get<0>(GetInput());
  auto& columns = std::get<1>(GetInput());

  if(rows <= 0 || columns <= 0) return false;

  auto& matrix = std::get<2>(GetInput());

  return (rows * columns == matrix.size()) && (GetOutput() == 0);
}

bool TabalaevAElemMatMinMPI::PreProcessingImpl() {
  return true;
}

bool TabalaevAElemMatMinMPI::RunImpl() {
  auto& matrix = std::get<2>(GetInput());

  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t matrix_size = matrix.size();
  size_t part_size = matrix_size / world_size;
  size_t remainder = matrix_size % world_size;
  

  std::vector<int> local_vec(world_rank == 0 ? (part_size + remainder) : part_size);

  if(world_rank == 0){
    std::copy(matrix.begin(), matrix.begin() + part_size + remainder, local_vec.begin());
    
    for(int i = 1; i < world_size; i++){
      size_t start = i * part_size + remainder;
      MPI_Send(matrix.data() + start, part_size, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_vec.data(), part_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  int local_minik = *std::min_element(local_vec.begin(), local_vec.end());

  int global_minik = 0;

  MPI_Allreduce(&local_minik, &global_minik, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_minik;

  return true;
}

bool TabalaevAElemMatMinMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_elem_mat_min
