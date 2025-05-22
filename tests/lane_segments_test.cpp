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

#include "lane_segments_test.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

namespace autoware::diffusion_planner::test
{

TEST_F(LaneSegmentsTest, ProcessSegmentToMatrix)
{
  auto segment_matrix = preprocess::process_segment_to_matrix(lane_segments_.front());

  ASSERT_EQ(segment_matrix.rows(), POINTS_PER_SEGMENT);  // Expect 3 rows (one for each point)
  ASSERT_EQ(segment_matrix.cols(), FULL_MATRIX_ROWS);    // Expect FULL_MATRIX_ROWS columns

  EXPECT_FLOAT_EQ(segment_matrix(0, X), 0.0);
  EXPECT_FLOAT_EQ(segment_matrix(POINTS_PER_SEGMENT - 1, X), 20.0);

  EXPECT_FLOAT_EQ(segment_matrix(0, LB_X), 0.0);
  EXPECT_FLOAT_EQ(segment_matrix(POINTS_PER_SEGMENT - 1, LB_X), 20.0);

  EXPECT_FLOAT_EQ(segment_matrix(0, RB_X), 0.0);
  EXPECT_FLOAT_EQ(segment_matrix(POINTS_PER_SEGMENT - 1, RB_X), 20.0);

  EXPECT_FLOAT_EQ(segment_matrix(0, SPEED_LIMIT), 30.0f);
}

TEST_F(LaneSegmentsTest, ProcessSegmentsToMatrix)
{
  preprocess::ColLaneIDMaps col_id_mapping;
  auto full_matrix = preprocess::process_segments_to_matrix(lane_segments_, col_id_mapping);

  ASSERT_EQ(
    full_matrix.rows(), POINTS_PER_SEGMENT);  // Expect 3 rows (one for each point in the segment)
  ASSERT_EQ(full_matrix.cols(), FULL_MATRIX_ROWS);  // Expect FULL_MATRIX_ROWS columns

  EXPECT_EQ(col_id_mapping.lane_id_to_matrix_col.size(), 1);  // Expect one lane ID mapping
  EXPECT_EQ(col_id_mapping.matrix_col_to_lane_id.size(), 1);  // Expect one row-to-lane mapping
}

TEST_F(LaneSegmentsTest, ComputeDistances)
{
  preprocess::ColLaneIDMaps col_id_mapping;
  auto input_matrix = preprocess::process_segments_to_matrix(lane_segments_, col_id_mapping);

  Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
  std::vector<preprocess::ColWithDistance> distances;

  preprocess::compute_distances(input_matrix, transform_matrix, distances, 10.0, 0.0, 100.0);

  ASSERT_EQ(distances.size(), 1);    // Expect one segment
  EXPECT_EQ(distances[0].index, 0);  // Expect the first segment index
  EXPECT_TRUE(distances[0].inside);  // Expect the segment to be inside the mask range
}

}  // namespace autoware::diffusion_planner::test
