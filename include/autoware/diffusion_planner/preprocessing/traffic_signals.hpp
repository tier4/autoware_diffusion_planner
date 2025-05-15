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

#ifndef AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__TRAFFIC_SIGNALS_HPP
#define AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__TRAFFIC_SIGNALS_HPP

#include "rclcpp/rclcpp.hpp"

#include <autoware_perception_msgs/msg/detail/traffic_light_group__struct.hpp>
#include <autoware_perception_msgs/msg/detail/traffic_light_group_array__struct.hpp>
#include <autoware_perception_msgs/msg/traffic_signal.hpp>

#include <lanelet2_traffic_rules/TrafficRules.h>

#include <map>

namespace autoware::diffusion_planner::preprocess
{
/**
 * @brief Represents a traffic signal with a timestamp.
 */
struct TrafficSignalStamped
{
  builtin_interfaces::msg::Time stamp;                      ///< Timestamp of the signal.
  autoware_perception_msgs::msg::TrafficLightGroup signal;  ///< Traffic light group.
};

void process_traffic_signals(
  const autoware_perception_msgs::msg::TrafficLightGroupArray::ConstSharedPtr msg,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_signal_id_map);

}  // namespace autoware::diffusion_planner::preprocess
#endif  // AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__TRAFFIC_SIGNALS_HPP
