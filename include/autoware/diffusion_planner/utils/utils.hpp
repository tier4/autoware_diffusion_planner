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

#ifndef AUTOWARE__DIFFUSION_PLANNER__UTILS_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__UTILS_HPP_

#include <Eigen/Dense>

#include "nav_msgs/msg/odometry.hpp"

#include <utility>

namespace autoware::diffusion_planner::utils
{

std::pair<Eigen::Matrix4f, Eigen::Matrix4f> get_transform_matrix(
  const nav_msgs::msg::Odometry & msg);

std::vector<float> create_float_data(const std::vector<int64_t> & shape, float fill = 0.1f);

}  // namespace autoware::diffusion_planner::utils
#endif  // AUTOWARE__DIFFUSION_PLANNER__UTILS_HPP_
