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

#ifndef AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__PREPROCESSING_UTILS_HPP
#define AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__PREPROCESSING_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::diffusion_planner::preprocess
{

using InputDataMap = std::unordered_map<std::string, std::vector<float>>;
using NormalizationMap =
  std::unordered_map<std::string, std::pair<std::vector<float>, std::vector<float>>>;

void normalize_input_data(InputDataMap & input_data_map, NormalizationMap & normalization_map);
}  // namespace autoware::diffusion_planner::preprocess
#endif  // AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__PREPROCESSING_UTILS_HPP
