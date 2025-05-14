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

#ifndef AUTOWARE__DIFFUSION_PLANNER__POSPROCESSING__POSPROCESSING_UTILS_HPP
#define AUTOWARE__DIFFUSION_PLANNER__POSPROCESSING__POSPROCESSING_UTILS_HPP

#include "onnxruntime_cxx_api.h"
#include "rclcpp/rclcpp.hpp"

#include <Eigen/Dense>

#include <autoware_planning_msgs/msg/trajectory.hpp>

#include <cassert>
#include <vector>

namespace autoware::diffusion_planner::postprocessing
{
using autoware_planning_msgs::msg::Trajectory;

void transform_output_matrix(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation = true);

Trajectory create_trajectory(
  std::vector<Ort::Value> & predictions, const rclcpp::Time & stamp,
  const Eigen::Matrix4f & transform_ego_to_map);

}  // namespace autoware::diffusion_planner::postprocessing
#endif  // AUTOWARE__DIFFUSION_PLANNER__POSPROCESSING__POSPROCESSING_UTILS_HPP
