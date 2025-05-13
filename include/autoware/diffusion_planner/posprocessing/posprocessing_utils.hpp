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

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::diffusion_planner::posprocessing
{

void transform_output_matrix(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation = true);

}  // namespace autoware::diffusion_planner::posprocessing
#endif  // AUTOWARE__DIFFUSION_PLANNER__POSPROCESSING__POSPROCESSING_UTILS_HPP
