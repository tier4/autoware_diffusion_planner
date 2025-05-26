// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/diffusion_planner/postprocessing/postprocessing_utils.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <vector>

namespace autoware::diffusion_planner::test
{
using autoware_planning_msgs::msg::Trajectory;

TEST(PostprocessingUtilsTest, TransformOutputMatrixTranslation)
{
  Eigen::MatrixXf output_matrix = Eigen::MatrixXf::Zero(4, OUTPUT_T);
  output_matrix.block<2, 5>(0, 0).setOnes();
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 2.0f;
  transform(1, 3) = -3.0f;

  postprocessing::transform_output_matrix(transform, output_matrix, 0, 0, true);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(output_matrix(0, i), 3.0f);   // 1 + 2
    EXPECT_FLOAT_EQ(output_matrix(1, i), -2.0f);  // 1 - 3
  }
}

TEST(PostprocessingUtilsTest, TransformOutputMatrixNoTranslation)
{
  Eigen::MatrixXf output_matrix = Eigen::MatrixXf::Zero(4, OUTPUT_T);
  output_matrix.block<2, 5>(2, 0).setOnes();
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 2.0f;
  transform(1, 3) = -3.0f;

  postprocessing::transform_output_matrix(transform, output_matrix, 0, 2, false);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(output_matrix(2, i), 1.0f);
    EXPECT_FLOAT_EQ(output_matrix(3, i), 1.0f);
  }
}

TEST(PostprocessingUtilsTest, GetTensorDataCopiesData)
{
  std::vector<float> data{1, 2, 3, 4, 5, 6};
  std::vector<int64_t> shape{1, 1, 2, 3};
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value ort_value =
    Ort::Value::CreateTensor<float>(mem_info, data.data(), data.size(), shape.data(), shape.size());

  auto mat = postprocessing::get_tensor_data(ort_value);
  ASSERT_EQ(mat.rows(), 2);
  ASSERT_EQ(mat.cols(), 3);
  EXPECT_FLOAT_EQ(mat(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(mat(1, 2), 6.0f);
}

TEST(PostprocessingUtilsTest, GetPredictionMatrixTransformsCorrectly)
{
  constexpr int rows = 3;
  constexpr int cols = OUTPUT_T;  // Use the actual OUTPUT_T value
  std::vector<float> data(rows * cols, 0.0f);
  // Fill with some values for checking
  for (int i = 0; i < rows * cols; ++i) data[i] = static_cast<float>(i);

  std::vector<int64_t> shape{1, 1, rows, cols};
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value ort_value =
    Ort::Value::CreateTensor<float>(mem_info, data.data(), data.size(), shape.data(), shape.size());

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 3) = 1.0f;
  transform(1, 3) = 2.0f;

  auto mat = postprocessing::get_prediction_matrix(ort_value, transform, 0, 0);
  ASSERT_EQ(mat.rows(), rows);
  ASSERT_EQ(mat.cols(), cols);
  // Optionally, check a few values to ensure transformation occurred
}

TEST(PostprocessingUtilsTest, GetTrajectoryFromPredictionMatrixWorks)
{
  Eigen::MatrixXf prediction_matrix(3, 4);
  prediction_matrix << 0, 0, 1, 0, 1, 0, 1, 1, 2, 0, 1, 0;
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  rclcpp::Time stamp(123, 0);

  auto traj =
    postprocessing::get_trajectory_from_prediction_matrix(prediction_matrix, transform, stamp);
  ASSERT_EQ(traj.points.size(), 3);
  EXPECT_FLOAT_EQ(traj.points[0].pose.position.x, 0.0f);
  EXPECT_FLOAT_EQ(traj.points[1].pose.position.x, 1.0f);
  EXPECT_GT(traj.points[1].longitudinal_velocity_mps, 0.0f);
}

TEST(PostprocessingUtilsTest, CreateTrajectoryAndMultipleTrajectories)
{
  constexpr int batch = 1;
  constexpr int agent = 2;
  constexpr int rows = 2;
  constexpr int cols = OUTPUT_T;
  std::vector<float> data(batch * agent * rows * cols, 0.0f);
  // Fill with some values for checking
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);

  std::vector<int64_t> shape{batch, agent, rows, cols};
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value ort_value =
    Ort::Value::CreateTensor<float>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  rclcpp::Time stamp(123, 0);

  auto traj = postprocessing::create_trajectory(ort_value, stamp, transform, 0, 0);
  ASSERT_EQ(traj.points.size(), rows);

  auto trajs = postprocessing::create_multiple_trajectories(ort_value, stamp, transform, 0, 0);
  ASSERT_EQ(trajs.size(), agent);
}

TEST(PostprocessingUtilsTest, ToTrajectoriesMsgPopulatesFields)
{
  using UUID = unique_identifier_msgs::msg::UUID;
  Trajectory traj;
  traj.header.frame_id = "map";
  traj.points.resize(2);
  UUID uuid;
  std::fill(uuid.uuid.begin(), uuid.uuid.end(), 42);

  auto msg = postprocessing::to_trajectories_msg(traj, uuid, "test_generator");
  ASSERT_EQ(msg.trajectories.size(), 1);
  ASSERT_EQ(msg.generator_info.size(), 1);
  EXPECT_EQ(msg.trajectories[0].header.frame_id, "map");
  EXPECT_EQ(msg.generator_info[0].generator_name.data, "test_generator");
}

}  // namespace autoware::diffusion_planner::test
