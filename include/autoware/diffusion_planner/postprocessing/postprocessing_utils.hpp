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

#include "autoware/diffusion_planner/conversion/agent.hpp"
#include "onnxruntime_cxx_api.h"
#include "rclcpp/rclcpp.hpp"

#include <Eigen/Dense>

#include <autoware_new_planning_msgs/msg/trajectories.hpp>
#include <autoware_perception_msgs/msg/detail/object_classification__struct.hpp>
#include <autoware_perception_msgs/msg/detail/predicted_object__struct.hpp>
#include <autoware_perception_msgs/msg/predicted_object.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_planning_msgs/msg/detail/trajectory__struct.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace autoware::diffusion_planner::postprocessing
{
using autoware_new_planning_msgs::msg::Trajectories;
using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_perception_msgs::msg::PredictedPath;
using autoware_planning_msgs::msg::Trajectory;
using unique_identifier_msgs::msg::UUID;

void transform_output_matrix(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation = true);

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> get_tensor_data(
  Ort::Value & prediction);

Eigen::MatrixXf get_prediction_matrix(
  Ort::Value & prediction, const Eigen::Matrix4f & transform_ego_to_map, const long batch = 0,
  const long agent = 0);

PredictedObjects create_predicted_objects(
  Ort::Value & prediction, const AgentData & ego_centric_agent_data, const rclcpp::Time & stamp,
  const Eigen::Matrix4f & transform_ego_to_map);

Trajectory get_trajectory_from_prediction_matrix(
  const Eigen::MatrixXf & prediction_matrix, const Eigen::Matrix4f & transform_ego_to_map,
  const rclcpp::Time & stamp);

Trajectory create_trajectory(
  Ort::Value & prediction, const rclcpp::Time & stamp, const Eigen::Matrix4f & transform_ego_to_map,
  long batch, long agent);

std::vector<Trajectory> create_multiple_trajectories(
  Ort::Value & prediction, const rclcpp::Time & stamp, const Eigen::Matrix4f & transform_ego_to_map,
  long start_batch, long start_agent);

Trajectories to_trajectories_msg(
  const Trajectory & trajectory, const UUID & generator_uuid, const std::string & generator_name);

}  // namespace autoware::diffusion_planner::postprocessing
#endif  // AUTOWARE__DIFFUSION_PLANNER__POSPROCESSING__POSPROCESSING_UTILS_HPP
