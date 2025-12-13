#pragma once

#include "tabalaev_a_linear_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace tabalaev_a_linear_topology {

class TabalaevALinearTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TabalaevALinearTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int GetLeft(int rank) const;
  int GetRight(int rank, int size) const;
  int GetDirection(int sender, int receiver) const;
  bool IsOnPath(int rank, int sender, int receiver, int direction) const;
  void ProcessSender(int direction, int left, int right, std::vector<int> &local_buff);
  void ProcessIntermediate(int direction, int left, int right, std::vector<int> &local_buff);
  void ProcessReceiver(int direction, int left, int right, std::vector<int> &local_buff);
};

}  // namespace tabalaev_a_linear_topology
