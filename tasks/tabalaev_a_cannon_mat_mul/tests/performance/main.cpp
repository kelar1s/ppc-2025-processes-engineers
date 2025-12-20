#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"
#include "tabalaev_a_cannon_mat_mul/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_cannon_mat_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tabalaev_a_cannon_mat_mul {

class TabalaevACannonMatMulPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    size_t rc = 8;
    size_t size = rc * rc;

    std::vector<double> A(rc * rc);
    std::vector<double> B(rc * rc);

    for (size_t i = 0; i < size; i++) {
      A[i] = static_cast<double>(1000 - static_cast<int>(i));
      B[i] = static_cast<double>(i) * static_cast<double>(1000) / static_cast<double>(size - 1);
    }

    input_data_ = std::make_tuple(rc, A, B);

    std::vector<double> C = MatMul(rc, A, B);
    expected_output_ = C;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  std::vector<double> MatMul(size_t N, const std::vector<double> &A, const std::vector<double> &B) {
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

    return C;
  }

 private:
  InType input_data_;
  std::vector<double> expected_output_ = {};
};

TEST_P(TabalaevACannonMatMulPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TabalaevACannonMatMulMPI, TabalaevACannonMatMulSEQ>(
    PPC_SETTINGS_tabalaev_a_cannon_mat_mul);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TabalaevACannonMatMulPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TabalaevACannonMatMulPerfTests, kGtestValues, kPerfTestName);

}  // namespace tabalaev_a_cannon_mat_mul
