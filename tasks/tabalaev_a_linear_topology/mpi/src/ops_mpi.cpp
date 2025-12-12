#include "tabalaev_a_linear_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <climits>
#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"

namespace tabalaev_a_linear_topology {

TabalaevALinearTopologyMPI::TabalaevALinearTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevALinearTopologyMPI::ValidationImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int sender = std::get<0>(GetInput());

  int validation = 1;
  if (world_rank == sender) {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    auto receiver = std::get<1>(GetInput());
    auto data = std::get<2>(GetInput());

    if ((sender < 0 || sender >= world_size) || (receiver < 0 || receiver >= world_size) || data.empty()) {
      validation = 0;
    }
  }

  MPI_Bcast(&validation, 1, MPI_INT, sender, MPI_COMM_WORLD);

  return validation != 0;
}

bool TabalaevALinearTopologyMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool TabalaevALinearTopologyMPI::RunImpl() {
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto sender = std::get<0>(GetInput());
  auto receiver = std::get<1>(GetInput());
  std::vector<int> data;

  if (sender == receiver) {
    GetOutput() = std::get<2>(GetInput());
    return true;
  }

  int left = (world_rank == 0 ? MPI_PROC_NULL : world_rank - 1);
  int right = (world_rank == world_size - 1 ? MPI_PROC_NULL : world_rank + 1);

  int direction = (sender < receiver ? 1 : -1);

  std::vector<int> local_buff;

  if (world_rank == sender) {
    data = std::get<2>(GetInput());

    int to = (direction == 1 ? right : left);

    int size = static_cast<int>(data.size());

    local_buff.resize(size);
    local_buff = data;

    MPI_Send(&size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
    MPI_Send(local_buff.data(), size, MPI_INT, to, 1, MPI_COMM_WORLD);
  }

  if (world_rank != sender && world_rank != receiver) {
    bool on_path = false;

    if (direction == 1) {
      if (world_rank > sender && world_rank < receiver) {
        on_path = true;
      }
    } else {
      if (world_rank < sender && world_rank > receiver) {
        on_path = true;
      }
    }

    if (on_path) {
      int from = (direction == 1 ? left : right);
      int to = (direction == 1 ? right : left);

      int size = 0;
      MPI_Recv(&size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      local_buff.resize(size);
      MPI_Recv(local_buff.data(), size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      MPI_Send(&size, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
      MPI_Send(local_buff.data(), size, MPI_INT, to, 1, MPI_COMM_WORLD);
    }
  }

  if (world_rank == receiver) {
    int from = (direction == 1 ? left : right);

    int size = 0;
    MPI_Recv(&size, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_buff.resize(size);
    MPI_Recv(local_buff.data(), size, MPI_INT, from, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int data_size = (world_rank == receiver) ? static_cast<int>(local_buff.size()) : 0;
  MPI_Bcast(&data_size, 1, MPI_INT, receiver, MPI_COMM_WORLD);

  if (world_rank != receiver) {
    local_buff.resize(data_size);
  }
  MPI_Bcast(local_buff.data(), data_size, MPI_INT, receiver, MPI_COMM_WORLD);

  GetOutput() = local_buff;

  return true;
}

bool TabalaevALinearTopologyMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_linear_topology
