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

#ifndef AUTOWARE__DIFFUSION_PLANNER__MARKER_UTILS_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__MARKER_UTILS_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>

#include <std_msgs/msg/detail/color_rgba__struct.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <array>
#include <string>
#include <vector>

namespace autoware::diffusion_planner::utils
{
using rclcpp::Duration;
using rclcpp::Time;
using std_msgs::msg::ColorRGBA;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

ColorRGBA get_traffic_light_color(float g, float y, float r, const ColorRGBA & original_color);

MarkerArray create_lane_marker(
  const std::vector<float> & lane_vector, const std::vector<long> & shape, const Time & stamp,
  const rclcpp::Duration & lifetime, const std::array<float, 4> colors = {0.0f, 1.0f, 0.0f, 0.8f},
  const std::string & frame_id = "base_link", const bool set_traffic_light_color = false);
}  // namespace autoware::diffusion_planner::utils
#endif  // AUTOWARE__DIFFUSION_PLANNER__MARKER_UTILS_HPP_
