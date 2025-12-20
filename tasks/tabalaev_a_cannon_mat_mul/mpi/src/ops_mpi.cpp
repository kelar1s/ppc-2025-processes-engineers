#include "tabalaev_a_cannon_mat_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"

namespace tabalaev_a_cannon_mat_mul {

void LocalMatrixMultiply(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, int size) {
  for (int i = 0; i < size; ++i) {
    for (int k = 0; k < size; ++k) {
      double temp = A[i * size + k];
      for (int j = 0; j < size; ++j) {
        C[i * size + j] += temp * B[k * size + j];
      }
    }
  }
}

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
  GetOutput() = {};
  return true;
}

bool TabalaevACannonMatMulMPI::RunImpl() {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int N = 0;
  if (world_rank == 0) {
    N = static_cast<int>(std::get<0>(GetInput()));
  }
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int q = static_cast<int>(std::sqrt(world_size));
  int blockSize = N / q;

  int dims[2] = {q, q};
  int periods[2] = {1, 1};
  MPI_Comm gridComm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm);

  int coords[2];
  MPI_Cart_coords(gridComm, world_rank, 2, coords);
  int row = coords[0];
  int col = coords[1];

  MPI_Datatype blockType, resizedBlockType;
  MPI_Type_vector(blockSize, blockSize, N, MPI_DOUBLE, &blockType);
  MPI_Type_commit(&blockType);

  MPI_Type_create_resized(blockType, 0, sizeof(double), &resizedBlockType);
  MPI_Type_commit(&resizedBlockType);

  std::vector<double> localA(blockSize * blockSize);
  std::vector<double> localB(blockSize * blockSize);
  std::vector<double> localC(blockSize * blockSize, 0.0);

  std::vector<int> counts(world_size, 1);
  std::vector<int> displs(world_size);
  if (world_rank == 0) {
    auto &A = std::get<1>(GetInput());
    auto &B = std::get<2>(GetInput());

    for (int i = 0; i < q; ++i) {
      for (int j = 0; j < q; ++j) {
        displs[i * q + j] = i * N * blockSize + j * blockSize;
      }
    }

    MPI_Scatterv(A.data(), counts.data(), displs.data(), resizedBlockType, localA.data(), blockSize * blockSize,
                 MPI_DOUBLE, 0, gridComm);
    MPI_Scatterv(B.data(), counts.data(), displs.data(), resizedBlockType, localB.data(), blockSize * blockSize,
                 MPI_DOUBLE, 0, gridComm);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, resizedBlockType, localA.data(), blockSize * blockSize, MPI_DOUBLE, 0,
                 gridComm);
    MPI_Scatterv(nullptr, nullptr, nullptr, resizedBlockType, localB.data(), blockSize * blockSize, MPI_DOUBLE, 0,
                 gridComm);
  }

  int left, right, up, down;
  MPI_Cart_shift(gridComm, 1, 1, &right, &left);
  MPI_Cart_shift(gridComm, 0, 1, &down, &up);

  for (int i = 0; i < row; ++i) {
    MPI_Sendrecv_replace(localA.data(), blockSize * blockSize, MPI_DOUBLE, left, 0, right, 0, gridComm,
                         MPI_STATUS_IGNORE);
  }
  for (int i = 0; i < col; ++i) {
    MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_DOUBLE, up, 1, down, 1, gridComm, MPI_STATUS_IGNORE);
  }

  for (int k = 0; k < q; ++k) {
    LocalMatrixMultiply(localA, localB, localC, blockSize);
    MPI_Sendrecv_replace(localA.data(), blockSize * blockSize, MPI_DOUBLE, left, 0, right, 0, gridComm,
                         MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(localB.data(), blockSize * blockSize, MPI_DOUBLE, up, 1, down, 1, gridComm, MPI_STATUS_IGNORE);
  }

  std::vector<double> globalResult(N * N);

  MPI_Gatherv(localC.data(), blockSize * blockSize, MPI_DOUBLE, globalResult.data(), counts.data(), displs.data(),
              resizedBlockType, 0, gridComm);

  MPI_Bcast(globalResult.data(), N * N, MPI_DOUBLE, 0, gridComm);

  GetOutput() = globalResult;

  MPI_Type_free(&resizedBlockType);
  MPI_Type_free(&blockType);
  MPI_Comm_free(&gridComm);

  return true;
}

bool TabalaevACannonMatMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tabalaev_a_cannon_mat_mul
