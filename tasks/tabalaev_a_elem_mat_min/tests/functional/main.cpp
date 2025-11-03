#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tabalaev_a_elem_mat_min/common/include/common.hpp"
#include "tabalaev_a_elem_mat_min/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_elem_mat_min/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_elem_mat_min {

class TabalaevAElemMatMinFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int minik = std::get<2>(test_param);
    std::string str_minik = (minik < 0) ? "minus" + std::to_string(-minik) : std::to_string(minik);
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "_min" + str_minik + "_" + std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    std::srand(static_cast<unsigned>(time(nullptr)));
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows = std::get<0>(params);
    int columns = std::get<1>(params);
    int minik = std::get<2>(params);

    std::vector<int> matrix(rows * columns);
    for(int& elem: matrix){
      elem = minik + std::rand() % (250 - minik + 1);
    }
    matrix[std::rand() % matrix.size()] = minik;
    input_data_ = std::make_tuple(static_cast<size_t>(rows), static_cast<size_t>(columns), matrix);
    expected_minik_ = minik;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_minik_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_minik_ = 0;
};

namespace {

TEST_P(TabalaevAElemMatMinFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(3, 3, -15, "Small_matrix"),
    std::make_tuple(5, 5, 32, "Medium_matrix"),
    std::make_tuple(10, 10, -41, "Large_matrix")
  };

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TabalaevAElemMatMinMPI, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_elem_mat_min),
                   ppc::util::AddFuncTask<TabalaevAElemMatMinSEQ, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_elem_mat_min));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TabalaevAElemMatMinFuncTests::PrintFuncTestName<TabalaevAElemMatMinFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TabalaevAElemMatMinFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tabalaev_a_elem_mat_min
