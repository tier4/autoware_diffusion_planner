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

#ifndef AUTOWARE__DIFFUSION_PLANNER__DIMENSIONS_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__DIMENSIONS_HPP_

#include <array>
#include <vector>

constexpr long POINTS_PER_LANE_SEGMENT = 20;  //!< Number of points in each lane segment.
constexpr long SEGMENT_POINT_DIM = 12;        // Dimension of a lane segment point
// Number of columns in a segment matrix
// (X,Y,dX,dY,LeftBoundX,LeftBoundY,RightBoundX,RightBoundX,TrafficLightEncoding(Dim4),Speed Limit)
constexpr long FULL_MATRIX_COLS = 14;
static constexpr long OUTPUT_T = 80;  // Output timestamp number

const std::vector<long> EGO_CURRENT_STATE_SHAPE = {1, 10};
const std::vector<long> NEIGHBOR_SHAPE = {1, 32, 21, 11};
const std::vector<long> LANE_HAS_SPEED_LIMIT_SHAPE = {1, 70, 1};
const std::vector<long> STATIC_OBJECTS_SHAPE = {1, 5, 10};
const std::vector<long> LANES_SHAPE = {1, 70, 20, 12};
const std::vector<long> LANES_SPEED_LIMIT_SHAPE = {1, 70, 1};
const std::vector<long> ROUTE_LANES_SHAPE = {1, 25, 20, 12};

#endif  // AUTOWARE__DIFFUSION_PLANNER__DIMENSIONS_HPP_
