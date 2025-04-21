// Copyright 2024 TIER IV, Inc.
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

#include "autoware/diffusion_planner/diffusion_planner_node.hpp"

namespace autoware::diffusion_planner
{
DiffusionPlanner::DiffusionPlanner(const rclcpp::NodeOptions & options)
: Node("diffusion_planner", options)
{
  // Initialize the node
  rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("diffusion_planner", options);
  pub_trajectory_ = node->create_publisher<Trajectory>("~/output/trajectory", 1);
  debug_processing_time_detail_pub_ = node->create_publisher<autoware_utils::ProcessingTimeDetail>(
    "~/debug/processing_time_detail_ms", 1);
  time_keeper_ = std::make_shared<autoware_utils::TimeKeeper>(debug_processing_time_detail_pub_);
  const auto planning_hz = 10.0;
  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Rate(planning_hz).period(),
    std::bind(&DiffusionPlanner::on_timer, this));
}
void DiffusionPlanner::on_timer()
{
  // Timer callback function
  RCLCPP_INFO(get_logger(), "Diffusion Planner Timer Callback");
}

void DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
