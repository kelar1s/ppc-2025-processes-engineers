#include <gtest/gtest.h>

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
    const double epsilon = 1e-6;
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

namespace {

TEST_P(TabalaevACannonMatMulFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(2, 100, "Small_matrix"),
    std::make_tuple(4, 1000, "Medium_matrix"),
    std::make_tuple(8, 10000, "Large_matrix"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TabalaevACannonMatMulMPI, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_cannon_mat_mul),
    ppc::util::AddFuncTask<TabalaevACannonMatMulSEQ, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_cannon_mat_mul));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TabalaevACannonMatMulFuncTests::PrintFuncTestName<TabalaevACannonMatMulFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TabalaevACannonMatMulFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tabalaev_a_cannon_mat_mul
