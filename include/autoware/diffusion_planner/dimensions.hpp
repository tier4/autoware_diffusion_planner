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

namespace autoware::diffusion_planner
{
constexpr long SEGMENT_POINT_DIM = 12;   // Dimension of a lane segment point
constexpr long POINTS_PER_SEGMENT = 20;  //!< Number of points in each lane segment.
// Number of columns in a segment matrix
// (X,Y,dX,dY,LeftBoundX,LeftBoundY,RightBoundX,RightBoundX,TrafficLightEncoding(Dim4),Speed Limit)
constexpr long FULL_MATRIX_COLS = 14;
constexpr long TRAFFIC_LIGHT_ONE_HOT_DIM = 4;

// Index for each field
constexpr long X = 0;
constexpr long Y = 1;
constexpr long dX = 2;
constexpr long dY = 3;
constexpr long LB_X = 4;
constexpr long LB_Y = 5;
constexpr long RB_X = 6;
constexpr long RB_Y = 7;
constexpr long TRAFFIC_LIGHT = 8;
constexpr long TRAFFIC_LIGHT_GREEN = 8;
constexpr long TRAFFIC_LIGHT_YELLOW = 9;
constexpr long TRAFFIC_LIGHT_RED = 10;
constexpr long TRAFFIC_LIGHT_WHITE = 11;
constexpr long SPEED_LIMIT = 12;
constexpr long LANE_ID = 13;

static constexpr long OUTPUT_T = 80;  // Output timestamp number

const std::vector<long> EGO_CURRENT_STATE_SHAPE = {1, 10};
const std::vector<long> NEIGHBOR_SHAPE = {1, 32, 21, 11};
const std::vector<long> LANE_HAS_SPEED_LIMIT_SHAPE = {1, 70, 1};
const std::vector<long> STATIC_OBJECTS_SHAPE = {1, 5, 10};
const std::vector<long> LANES_SHAPE = {1, 70, 20, 12};
const std::vector<long> LANES_SPEED_LIMIT_SHAPE = {1, 70, 1};
const std::vector<long> ROUTE_LANES_SHAPE = {1, 25, 20, 12};
}  // namespace autoware::diffusion_planner
#endif  // AUTOWARE__DIFFUSION_PLANNER__DIMENSIONS_HPP_
