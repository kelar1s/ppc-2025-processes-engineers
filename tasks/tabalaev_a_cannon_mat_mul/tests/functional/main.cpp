#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tabalaev_a_cannon_mat_mul/common/include/common.hpp"
#include "tabalaev_a_cannon_mat_mul/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_cannon_mat_mul/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_cannon_mat_mul {

class TabalaevACannonMatMulFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param) + "_" + std::to_string(std::get<0>(test_param)) + "x" +
           std::to_string(std::get<0>(test_param)) + "_Elems_Up_To_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    size_t rc = std::get<0>(params);
    size_t size = rc * rc;

    int upTo = std::get<1>(params);

    std::vector<double> A(rc * rc);
    std::vector<double> B(rc * rc);

    for (size_t i = 0; i < size; i++) {
      A[i] = static_cast<double>(upTo - static_cast<int>(i));
      B[i] = static_cast<double>(i) * static_cast<double>(upTo) / static_cast<double>(size - 1);
    }

    input_data_ = std::make_tuple(rc, A, B);
    std::vector<double> C = MatMul(rc, A, B);
    expected_output_ = C;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_output_.size() != output_data.size()) {
      return false;
    }
    const double epsilon = 1e-7;
    for (size_t i = 0; i < expected_output_.size(); ++i) {
      if (std::abs(expected_output_[i] - output_data[i]) > epsilon) {
        return false;
      }
    }
    return true;
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
  OutType expected_output_;
};

class TabalaevACannonSeqTests : public TabalaevACannonMatMulFuncTests {};

class TabalaevACannonMpiTests : public TabalaevACannonMatMulFuncTests {
 protected:
  void SetUp() override {
    if (!ppc::util::IsUnderMpirun()) {
      std::cerr << "Is not under mpi run\n";
      GTEST_SKIP();
    }
    TabalaevACannonMatMulFuncTests::SetUp();
  }
};

namespace {

TEST_P(TabalaevACannonSeqTests, SeqTest) {
  ExecuteTest(GetParam());
}

TEST_P(TabalaevACannonMpiTests, MpiTest) {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int q = static_cast<int>(std::sqrt(world_size));
  TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  int N = std::get<0>(params);
  if (q * q != world_size && N % q == 0) {
    ExecuteTest(GetParam());
  }
  std::cerr << "The conditions for matrix multiplication using Cannon's method are not met!\n";
}

const std::array<TestType, 6> kSeqParams = {
    std::make_tuple(2, 100, "Small_matrix"),   std::make_tuple(3, 150, "Medium_small_matrix"),
    std::make_tuple(4, 200, "Medium_matrix"),  std::make_tuple(6, 500, "Medium_large_matrix"),
    std::make_tuple(10, 1000, "Large_matrix"), std::make_tuple(12, 2000, "Extra_large_matrix")};

const std::array<TestType, 6> kMpiParams = {
    std::make_tuple(2, 100, "Small_matrix"),  std::make_tuple(3, 150, "Medium_small_matrix"),
    std::make_tuple(4, 200, "Medium_matrix"), std::make_tuple(6, 500, "Medium_large_matrix"),
    std::make_tuple(8, 1000, "Large_matrix"), std::make_tuple(12, 2000, "Extra_large_matrix")};

const auto kSeqTasks =
    ppc::util::AddFuncTask<TabalaevACannonMatMulSEQ, InType>(kSeqParams, PPC_SETTINGS_tabalaev_a_cannon_mat_mul);
const auto kMpiTasks =
    ppc::util::AddFuncTask<TabalaevACannonMatMulMPI, InType>(kMpiParams, PPC_SETTINGS_tabalaev_a_cannon_mat_mul);

const auto kFuncTestName = TabalaevACannonMatMulFuncTests::PrintFuncTestName<TabalaevACannonMatMulFuncTests>;

INSTANTIATE_TEST_SUITE_P(SeqTests, TabalaevACannonSeqTests, ppc::util::ExpandToValues(kSeqTasks), kFuncTestName);

INSTANTIATE_TEST_SUITE_P(MpiTests, TabalaevACannonMpiTests, ppc::util::ExpandToValues(kMpiTasks), kFuncTestName);

}  // namespace

}  // namespace tabalaev_a_cannon_mat_mul
