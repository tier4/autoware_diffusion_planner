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

#ifndef AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__LANE_SEGMENTS_HPP
#define AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__LANE_SEGMENTS_HPP

#include "autoware/diffusion_planner/conversion/lanelet.hpp"
#include "autoware/diffusion_planner/dimensions.hpp"
#include "autoware/diffusion_planner/preprocessing/traffic_signals.hpp"
#include "autoware/traffic_light_utils/traffic_light_utils.hpp"

#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>

#include <Eigen/src/Core/Matrix.h>
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/CompoundPolygon.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/utility/Optional.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::diffusion_planner::preprocess
{
using autoware_planning_msgs::msg::LaneletRoute;
/**
 * @brief Represents a row index with its associated distance and whether it is inside a mask range.
 */
struct RowWithDistance
{
  long index;              //!< Row index in the input matrix.
  float distance_squared;  //!< Squared distance from the center.
  bool inside;             //!< Whether the row is within the mask range.
};

struct RowLaneIDMaps
{
  std::map<lanelet::Id, long> lane_id_to_matrix_row;
  std::map<long, lanelet::Id> matrix_row_to_lane_id;
};

/**
 * @brief Extracts route segments from the map lane segments matrix and transforms them to
 * ego-centric coordinates.
 * @param map_lane_segments_matrix Matrix containing lane segment data in map coordinates.
 * @param map_to_ego_transform Transformation matrix to convert map coordinates to ego-centric
 * coordinates.
 * @param route_ptr_ Shared pointer to the route containing the segments to extract.
 * @param segment_row_indices Map of segment IDs to their corresponding row indices in the lane
 * segments matrix.
 * @param center_x X-coordinate of the ego vehicle's center.
 * @param center_y Y-coordinate of the ego vehicle's center.
 * @return A flattened vector containing the transformed route segments in ego-centric coordinates.
 */
std::vector<float> get_route_segments(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  LaneletRoute::ConstSharedPtr route_ptr_, const RowLaneIDMaps & row_id_mapping,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y);

/**
 * @brief Extracts lane tensor data from ego-centric lane segments.
 *
 * @param lane_segments_matrix Matrix containing ego-centric lane segment data.
 * @return A flattened vector of lane tensor data.
 */
std::vector<float> extract_lane_tensor_data(const Eigen::MatrixXf & lane_segments_matrix);

/**
 * @brief Extracts lane speed tensor data from ego-centric lane segments.
 *
 * @param lane_segments_matrix Matrix containing ego-centric lane segment data.
 * @return A flattened vector of lane speed tensor data.
 */
std::vector<float> extract_lane_speed_tensor_data(const Eigen::MatrixXf & lane_segments_matrix);

/**
 * @brief Processes multiple lane segments and converts them into a single matrix.
 *
 * @param lane_segments Vector of lane segments to process.
 * @param segment_row_indices Map to store the starting row index of each segment in the resulting
 * matrix.
 * @return A matrix containing the processed lane segment data.
 * @throws std::runtime_error If any segment matrix does not have the expected number of rows.
 */
Eigen::MatrixXf process_segments_to_matrix(
  const std::vector<LaneSegment> & lane_segments, RowLaneIDMaps & row_id_mapping);

/**
 * @brief Processes a single lane segment and converts it into a matrix representation.
 *
 * @param segment The lane segment to process.
 * @return A matrix containing the processed lane segment data, or an empty matrix if the segment is
 * invalid.
 */
Eigen::MatrixXf process_segment_to_matrix(const LaneSegment & segment);

/**
 * @brief Computes distances of lane segments from a center point and stores the results.
 *
 * @param input_matrix Input matrix containing lane segment data.
 * @param transform_matrix Transformation matrix to apply to the points.
 * @param distances Output vector to store row indices, distances, and mask inclusion.
 * @param center_x X-coordinate of the center point.
 * @param center_y Y-coordinate of the center point.
 * @param mask_range Range within which rows are considered "inside" the mask.
 */
void compute_distances(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  std::vector<RowWithDistance> & distances, float center_x, float center_y,
  float mask_range = 100.0);

/**
 * @brief Sorts the rows by their squared distances in ascending order.
 *
 * @param distances Vector of rows with distances to be sorted.
 */
inline void sort_indices_by_distance(std::vector<RowWithDistance> & distances)
{
  std::sort(distances.begin(), distances.end(), [&](auto & a, auto & b) {
    return a.distance_squared < b.distance_squared;
  });
}

void add_traffic_light_one_hot_encoding_to_segment(
  [[maybe_unused]] Eigen::MatrixXf & segment_matrix, const RowLaneIDMaps & row_id_mapping,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, const long row_idx,
  [[maybe_unused]] const long col_counter);

Eigen::RowVector4f get_traffic_signal_row_vector(
  const autoware_perception_msgs::msg::TrafficLightGroup & signal);

/**
 * @brief Transforms selected columns of the output matrix using a transformation matrix.
 *
 * @param transform_matrix Transformation matrix to apply.
 * @param output_matrix Matrix to transform (in-place).
 * @param column_idx Index of the column to transform.
 * @param row_idx Index of the row to transform.
 * @param do_translation Whether to apply translation during the transformation.
 */
void transform_selected_cols(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation = true);

/**
 * @brief Transforms and selects rows from the input matrix based on distances.
 *
 * @param input_matrix Input matrix containing lane segment data.
 * @param transform_matrix Transformation matrix to apply to the points.
 * @param distances Vector of rows with distances, used to select rows.
 * @param m Maximum number of rows to select.
 * @return Transformed matrix containing the selected rows.
 */
Eigen::MatrixXf transform_points_and_add_traffic_info(
  const Eigen::MatrixXf & input_matrix, [[maybe_unused]] const Eigen::Matrix4f & transform_matrix,
  [[maybe_unused]] const std::vector<RowWithDistance> & distances,
  [[maybe_unused]] const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, long m);

/**
 * @brief Transforms and selects rows from the input matrix based on proximity to a center point.
 *
 * @param input_matrix Input matrix containing lane segment data.
 * @param transform_matrix Transformation matrix to apply to the points.
 * @param center_x X-coordinate of the center point.
 * @param center_y Y-coordinate of the center point.
 * @param m Maximum number of rows to select.
 * @return Transformed matrix containing the selected rows.
 * @throws std::invalid_argument If input_matrix dimensions are not correct or if m <= 0.
 */
Eigen::MatrixXf transform_and_select_rows(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  const RowLaneIDMaps & row_id_mapping,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y,
  long m);
}  // namespace autoware::diffusion_planner::preprocess

#endif  // AUTOWARE__DIFFUSION_PLANNER__PREPROCESSING__LANE_SEGMENTS_HPP
