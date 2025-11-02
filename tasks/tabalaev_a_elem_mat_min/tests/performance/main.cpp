#include <gtest/gtest.h>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"
#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_elem_mat_min/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tabalaev_a_elem_mat_min {

class TabalaevAElemMatMinPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TabalaevAElemMatMinPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TabalaevAElemMatMinMPI, TabalaevAElemMatMinSEQ>(PPC_SETTINGS_tabalaev_a_elem_mat_min);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TabalaevAElemMatMinPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TabalaevAElemMatMinPerfTests, kGtestValues, kPerfTestName);

}  // namespace tabalaev_a_elem_mat_min
