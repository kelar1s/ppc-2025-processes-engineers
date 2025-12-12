#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tabalaev_a_linear_topology/common/include/common.hpp"
#include "tabalaev_a_linear_topology/mpi/include/ops_mpi.hpp"
#include "tabalaev_a_linear_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_linear_topology {

class TabalaevALinearTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "Sender_" + std::to_string(std::get<0>(test_param)) + "_Receiver_" +
           std::to_string(std::get<1>(test_param)) + "_DataSize_" + std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int sender = std::get<0>(params);
    int receiver = std::get<1>(params);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      if (world_size < (std::max(sender, receiver) + 1)) {
        GTEST_SKIP() << "Skipping test: not enough processes";
      }
    }

    int size = std::get<2>(params);

    std::vector<int> data(size);

    std::uniform_int_distribution<int> dist(0, 250);

    for (int &elem : data) {
      elem = dist(gen_);
    }

    input_data_ = std::make_tuple(sender, receiver, data);
    expected_output_ = std::vector<int>(data);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = {};
  std::mt19937 gen_{12345};
};

namespace {

TEST_P(TabalaevALinearTopologyFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(0, 0, 50, "From 0 to 0"), std::make_tuple(0, 1, 50, "From 0 to 1"),
    std::make_tuple(1, 0, 50, "From 1 to 0"), std::make_tuple(3, 3, 50, "From 3 to 3"),
    std::make_tuple(0, 4, 50, "From 0 to 4"), std::make_tuple(4, 0, 50, "From 4 to 0"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TabalaevALinearTopologyMPI, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_linear_topology),
    ppc::util::AddFuncTask<TabalaevALinearTopologySEQ, InType>(kTestParam, PPC_SETTINGS_tabalaev_a_linear_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TabalaevALinearTopologyFuncTests::PrintFuncTestName<TabalaevALinearTopologyFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, TabalaevALinearTopologyFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tabalaev_a_linear_topology
