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

#include "autoware/diffusion_planner/dimensions.hpp"
#include "autoware/diffusion_planner/postprocessing/postprocessing_utils.hpp"

#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <rclcpp/time.hpp>

namespace autoware::diffusion_planner::postprocessing
{
using autoware_planning_msgs::msg::TrajectoryPoint;

void transform_output_matrix(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation)
{
  Eigen::Matrix<float, 4, OUTPUT_T> xy_block = Eigen::Matrix<float, 4, OUTPUT_T>::Zero();
  xy_block.block<2, OUTPUT_T>(0, 0) =
    output_matrix.block<2, OUTPUT_T>(row_idx, column_idx * OUTPUT_T);
  xy_block.row(3) = do_translation ? Eigen::Matrix<float, 1, OUTPUT_T>::Ones()
                                   : Eigen::Matrix<float, 1, OUTPUT_T>::Zero();

  Eigen::Matrix<float, 4, OUTPUT_T> transformed_block = transform_matrix * xy_block;
  output_matrix.block<2, OUTPUT_T>(row_idx, column_idx * OUTPUT_T) =
    transformed_block.block<2, OUTPUT_T>(0, 0);
};

Trajectory create_trajectory(
  std::vector<Ort::Value> & predictions, const rclcpp::Time & stamp,
  const Eigen::Matrix4f & transform_ego_to_map)
{
  Trajectory trajectory;
  trajectory.header.stamp = stamp;
  trajectory.header.frame_id = "map";
  // one batch of predictions
  // TODO(Daniel): add batch support
  auto data = predictions[0].GetTensorMutableData<float>();
  const auto prediction_shape = predictions[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto num_of_dimensions = prediction_shape.size();

  // copy relevant part of data to Eigen matrix
  auto rows = prediction_shape[num_of_dimensions - 2];
  auto cols = prediction_shape[num_of_dimensions - 1];

  Eigen::MatrixXf prediction_matrix(rows, cols);

  // Fill matrix row-wise from data using Eigen::Map
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped_data(
    data, rows, cols);

  // Copy only the relevant part
  prediction_matrix = mapped_data;  // Copies first rows*cols elements row-wise
  prediction_matrix.transposeInPlace();
  postprocessing::transform_output_matrix(transform_ego_to_map, prediction_matrix, 0, 0, true);
  postprocessing::transform_output_matrix(transform_ego_to_map, prediction_matrix, 0, 2, false);
  prediction_matrix.transposeInPlace();

  // TODO(Daniel): check there is no issue with the speed of 1st point (index 0)
  constexpr double dt = 0.1f;
  double prev_x = 0.;
  double prev_y = 0.;
  for (long row = 0; row < prediction_matrix.rows(); ++row) {
    TrajectoryPoint p;
    p.pose.position.x = prediction_matrix(row, 0);
    p.pose.position.y = prediction_matrix(row, 1);
    p.pose.position.z = 0.0;
    auto yaw = std::atan2(prediction_matrix(row, 3), prediction_matrix(row, 2));
    yaw = static_cast<float>(autoware_utils::normalize_radian(yaw));
    p.pose.orientation = autoware_utils::create_quaternion_from_yaw(yaw);
    auto distance = std::hypot(p.pose.position.x - prev_x, p.pose.position.y - prev_y);
    p.longitudinal_velocity_mps = static_cast<float>(distance / dt);

    prev_x = p.pose.position.x;
    prev_y = p.pose.position.y;
    trajectory.points.push_back(p);
  }
  return trajectory;
}

}  // namespace autoware::diffusion_planner::postprocessing
