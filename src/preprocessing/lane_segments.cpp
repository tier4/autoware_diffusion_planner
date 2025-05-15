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

#include "autoware/diffusion_planner/preprocessing/lane_segments.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

#include <autoware_lanelet2_extension/regulatory_elements/road_marking.hpp>  // for lanelet::autoware::RoadMarking
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>

#include <Eigen/src/Core/Matrix.h>

#include <cmath>
#include <iostream>

namespace autoware::diffusion_planner::preprocess
{
void compute_distances(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  std::vector<RowWithDistance> & distances, float center_x, float center_y, float mask_range)
{
  const auto rows = input_matrix.rows();
  if (rows % POINTS_PER_SEGMENT != 0) {
    throw std::runtime_error("input matrix rows are not divisible by POINTS_PER_SEGMENT");
  }

  auto compute_squared_distance = [](float x, float y, const Eigen::Matrix4f & transform_matrix) {
    Eigen::Vector4f p(x, y, 0.0f, 1.0f);
    Eigen::Vector4f p_transformed = transform_matrix * p;
    return p_transformed.head<2>().squaredNorm();
  };

  distances.clear();
  distances.reserve(rows / POINTS_PER_SEGMENT);
  for (long i = 0; i < rows; i += POINTS_PER_SEGMENT) {
    // Directly access input matrix as raw memory
    float x = input_matrix.block(i, 0, POINTS_PER_SEGMENT, 1).mean();
    float y = input_matrix.block(i, 1, POINTS_PER_SEGMENT, 1).mean();
    bool inside =
      (x > center_x - mask_range * 1.1 && x < center_x + mask_range * 1.1 &&
       y > center_y - mask_range * 1.1 && y < center_y + mask_range * 1.1);

    const auto distance_squared = [&]() {
      float x_first = input_matrix(i, X);
      float y_first = input_matrix(i, Y);
      float x_last = input_matrix(i + POINTS_PER_SEGMENT - 1, X);
      float y_last = input_matrix(i + POINTS_PER_SEGMENT - 1, Y);
      float distance_squared_first = compute_squared_distance(x_first, y_first, transform_matrix);
      float distance_squared_last = compute_squared_distance(x_last, y_last, transform_matrix);
      return std::min(distance_squared_last, distance_squared_first);
    }();

    distances.push_back({i, distance_squared, inside});
  }
}

void transform_selected_cols(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation)
{
  Eigen::Matrix<float, 4, POINTS_PER_SEGMENT> xy_block =
    Eigen::Matrix<float, 4, POINTS_PER_SEGMENT>::Zero();
  xy_block.block<2, POINTS_PER_SEGMENT>(0, 0) =
    output_matrix.block<2, POINTS_PER_SEGMENT>(row_idx, column_idx * POINTS_PER_SEGMENT);
  xy_block.row(3) = do_translation ? Eigen::Matrix<float, 1, POINTS_PER_SEGMENT>::Ones()
                                   : Eigen::Matrix<float, 1, POINTS_PER_SEGMENT>::Zero();

  Eigen::Matrix<float, 4, POINTS_PER_SEGMENT> transformed_block = transform_matrix * xy_block;
  output_matrix.block<2, POINTS_PER_SEGMENT>(row_idx, column_idx * POINTS_PER_SEGMENT) =
    transformed_block.block<2, POINTS_PER_SEGMENT>(0, 0);
}

void add_traffic_light_one_hot_encoding_to_segment(
  [[maybe_unused]] Eigen::MatrixXf & segment_matrix, const RowLaneIDMaps & row_id_mapping,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, const long row_idx,
  [[maybe_unused]] const long col_counter)
{
  const auto lane_id_itr = row_id_mapping.matrix_row_to_lane_id.find(row_idx);
  if (lane_id_itr == row_id_mapping.matrix_row_to_lane_id.end()) {
    throw std::invalid_argument("Invalid lane row to lane id mapping");
    return;
  }
  const auto assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id_itr->second);
  auto tl_reg_elems = assigned_lanelet.regulatoryElementsAs<const lanelet::TrafficLight>();

  if (tl_reg_elems.empty()) {
    return;
  }
  const auto & tl_reg_elem = tl_reg_elems.front();
  const auto traffic_light_stamped_info_itr = traffic_light_id_map.find(tl_reg_elem->id());
  if (traffic_light_stamped_info_itr == traffic_light_id_map.end()) {
    return;
  }
  const auto & signal = traffic_light_stamped_info_itr->second.signal;

  Eigen::Vector4f traffic_light_one_hot_encoding = get_traffic_signal_row_vector(signal);
  Eigen::MatrixXf one_hot_encoding_matrix =
    traffic_light_one_hot_encoding.replicate(1, POINTS_PER_SEGMENT);
  segment_matrix.block<TRAFFIC_LIGHT_ONE_HOT_DIM, POINTS_PER_SEGMENT>(
    TRAFFIC_LIGHT, col_counter * POINTS_PER_SEGMENT) =
    one_hot_encoding_matrix.block<TRAFFIC_LIGHT_ONE_HOT_DIM, POINTS_PER_SEGMENT>(0, 0);
}

Eigen::RowVector4f get_traffic_signal_row_vector(
  const autoware_perception_msgs::msg::TrafficLightGroup & signal)
{
  const auto is_green = autoware::traffic_light_utils::hasTrafficLightCircleColor(
    signal, autoware_perception_msgs::msg::TrafficLightElement::GREEN);
  const auto is_amber = autoware::traffic_light_utils::hasTrafficLightCircleColor(
    signal, autoware_perception_msgs::msg::TrafficLightElement::AMBER);
  const auto is_red = autoware::traffic_light_utils::hasTrafficLightCircleColor(
    signal, autoware_perception_msgs::msg::TrafficLightElement::RED);

  const bool has_color = (is_green || is_amber || is_red);

  if (
    static_cast<float>(is_green) + static_cast<float>(is_amber) + static_cast<float>(is_red) >
    1.f) {
    throw std::invalid_argument("more than one traffic light");
    return {};
  }
  return {
    static_cast<float>(is_green), static_cast<float>(is_amber), static_cast<float>(is_red),
    static_cast<float>(!has_color)};
}

std::tuple<Eigen::MatrixXf, RowLaneIDMaps> transform_points_and_add_traffic_info(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  const std::vector<RowWithDistance> & distances, const RowLaneIDMaps & row_id_mapping,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, long m)
{
  if (input_matrix.cols() != FULL_MATRIX_COLS || input_matrix.rows() % POINTS_PER_SEGMENT != 0) {
    throw std::invalid_argument("input_matrix size mismatch");
  }

  const long n_total_segments = static_cast<long>(input_matrix.rows() / POINTS_PER_SEGMENT);
  const long num_segments = std::min(m, n_total_segments);

  Eigen::MatrixXf output_matrix(m * POINTS_PER_SEGMENT, FULL_MATRIX_COLS);
  output_matrix.setZero();
  output_matrix.transposeInPlace();  // helps to simplify the code below

  long col_counter = 0;
  long added_segments = 0;
  RowLaneIDMaps new_row_id_mapping;
  for (auto distance : distances) {
    if (!distance.inside) {
      continue;
    }
    const auto row_idx = distance.index;
    const auto lane_id = row_id_mapping.matrix_row_to_lane_id.find(row_idx);

    if (lane_id == row_id_mapping.matrix_row_to_lane_id.end()) {
      throw std::invalid_argument("input_matrix size mismatch");
    }

    // get POINTS_PER_SEGMENT rows corresponding to a single segment
    output_matrix.block<FULL_MATRIX_COLS, POINTS_PER_SEGMENT>(0, col_counter * POINTS_PER_SEGMENT) =
      input_matrix.block<POINTS_PER_SEGMENT, FULL_MATRIX_COLS>(row_idx, 0).transpose();

    add_traffic_light_one_hot_encoding_to_segment(
      output_matrix, row_id_mapping, traffic_light_id_map, lanelet_map_ptr, row_idx, col_counter);

    // transform the x and y coordinates
    transform_selected_cols(transform_matrix, output_matrix, col_counter, X);
    // the dx and dy coordinates do not require translation
    transform_selected_cols(transform_matrix, output_matrix, col_counter, dX, false);
    transform_selected_cols(transform_matrix, output_matrix, col_counter, LB_X);
    transform_selected_cols(transform_matrix, output_matrix, col_counter, RB_X);

    new_row_id_mapping.lane_id_to_matrix_row.emplace(
      lane_id->second, col_counter * POINTS_PER_SEGMENT);
    new_row_id_mapping.matrix_row_to_lane_id.emplace(
      col_counter * POINTS_PER_SEGMENT, lane_id->second);
    ++col_counter;
    ++added_segments;
    if (added_segments >= num_segments) {
      break;
    }
  }
  // subtract center from boundaries
  output_matrix.row(LB_X) = output_matrix.row(4) - output_matrix.row(X);
  output_matrix.row(LB_Y) = output_matrix.row(5) - output_matrix.row(Y);
  output_matrix.row(RB_X) = output_matrix.row(6) - output_matrix.row(X);
  output_matrix.row(RB_Y) = output_matrix.row(7) - output_matrix.row(Y);

  return {output_matrix.transpose(), new_row_id_mapping};
}

std::tuple<Eigen::MatrixXf, RowLaneIDMaps> transform_and_select_rows(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  const RowLaneIDMaps & row_id_mapping,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y,
  const long m)
{
  const auto n = input_matrix.rows();
  if (n == 0 || input_matrix.cols() != FULL_MATRIX_COLS || m <= 0) {
    throw std::invalid_argument(
      "Input matrix must have at least FULL_MATRIX_COLS columns and m must be greater than 0.");
    return {};
  }
  std::vector<RowWithDistance> distances;
  // Step 1: Compute distances
  compute_distances(input_matrix, transform_matrix, distances, center_x, center_y, 10000);
  // Step 2: Sort indices by distance
  sort_indices_by_distance(distances);
  // Step 3: Apply transformation to selected rows
  return transform_points_and_add_traffic_info(
    input_matrix, transform_matrix, distances, row_id_mapping, traffic_light_id_map,
    lanelet_map_ptr, m);
}

Eigen::MatrixXf process_segments_to_matrix(
  const std::vector<LaneSegment> & lane_segments, RowLaneIDMaps & row_id_mapping)
{
  if (lane_segments.empty()) {
    throw std::runtime_error("Empty lane segment data");
  }
  std::vector<Eigen::MatrixXf> all_segment_matrices;
  for (const auto & segment : lane_segments) {
    Eigen::MatrixXf segment_matrix = process_segment_to_matrix(segment);

    if (segment_matrix.rows() != POINTS_PER_SEGMENT) {
      throw std::runtime_error("Segment matrix rows not equal to POINTS_PER_SEGMENT");
    }
    all_segment_matrices.push_back(segment_matrix);
  }

  // Now allocate the full matrix
  const long rows = static_cast<long>(POINTS_PER_SEGMENT) * static_cast<long>(lane_segments.size());
  const long cols = all_segment_matrices[0].cols();
  Eigen::MatrixXf stacked_matrix(rows, cols);

  long current_row = 0;
  for (const auto & mat : all_segment_matrices) {
    stacked_matrix.middleRows(current_row, mat.rows()) = mat;
    const auto id = static_cast<int64_t>(mat(0, LANE_ID));
    row_id_mapping.lane_id_to_matrix_row.emplace(id, current_row);
    row_id_mapping.matrix_row_to_lane_id.emplace(current_row, id);
    current_row += POINTS_PER_SEGMENT;
  }
  return stacked_matrix;
}

Eigen::MatrixXf process_segment_to_matrix(const LaneSegment & segment)
{
  if (
    segment.polyline.is_empty() || segment.left_boundaries.empty() ||
    segment.right_boundaries.empty()) {
    return {};
  }
  const auto & centerlines = segment.polyline.waypoints();
  const auto & left_boundaries = segment.left_boundaries.front().waypoints();
  const auto & right_boundaries = segment.right_boundaries.front().waypoints();

  const size_t n_rows = centerlines.size();
  if (left_boundaries.size() != n_rows || right_boundaries.size() != n_rows) {
    return {};
  }

  Eigen::MatrixXf segment_data(n_rows, FULL_MATRIX_COLS);

  // Encode traffic light as one-hot
  Eigen::Vector4f traffic_light_vec = Eigen::Vector4f::Zero();
  switch (segment.traffic_light) {
    case 1:
      traffic_light_vec[2] = 1.0f;
      break;  // RED
    case 2:
      traffic_light_vec[1] = 1.0f;
      break;  // AMBER
    case 3:
      traffic_light_vec[0] = 1.0f;
      break;  // GREEN
    default:
      traffic_light_vec[3] = 1.0f;
      break;  // WHITE/UNKNOWN
  }

  // Build each row
  for (long i = 0; i < static_cast<long>(n_rows); ++i) {
    segment_data(i, X) = centerlines[i].x();
    segment_data(i, Y) = centerlines[i].y();
    segment_data(i, dX) =
      i < static_cast<long>(n_rows) - 1 ? centerlines[i + 1].x() - centerlines[i].x() : 0.0f;
    segment_data(i, dY) =
      i < static_cast<long>(n_rows) - 1 ? centerlines[i + 1].y() - centerlines[i].y() : 0.0f;
    segment_data(i, LB_X) = left_boundaries[i].x();
    segment_data(i, LB_Y) = left_boundaries[i].y();
    segment_data(i, RB_X) = right_boundaries[i].x();
    segment_data(i, RB_Y) = right_boundaries[i].y();
    segment_data.block<1, 4>(i, TRAFFIC_LIGHT) = traffic_light_vec.transpose();
    segment_data(i, SPEED_LIMIT) = segment.speed_limit_mph.value_or(0.0f);
    segment_data(i, LANE_ID) = static_cast<float>(segment.id);
  }

  return segment_data;
}

std::vector<float> extract_lane_tensor_data(const Eigen::MatrixXf & lane_segments_matrix)
{
  const auto total_lane_points = LANES_SHAPE[1] * POINTS_PER_SEGMENT;
  Eigen::MatrixXf lane_matrix(total_lane_points, SEGMENT_POINT_DIM);
  lane_matrix.block(0, 0, total_lane_points, SEGMENT_POINT_DIM) =
    lane_segments_matrix.block(0, 0, total_lane_points, SEGMENT_POINT_DIM);
  lane_matrix.transposeInPlace();
  return {lane_matrix.data(), lane_matrix.data() + lane_matrix.size()};
}

std::vector<float> extract_lane_speed_tensor_data(const Eigen::MatrixXf & lane_segments_matrix)
{
  const auto total_lane_points = LANES_SPEED_LIMIT_SHAPE[1];
  Eigen::MatrixXf lane_speed_matrix(total_lane_points, LANES_SPEED_LIMIT_SHAPE[2]);
  lane_speed_matrix.block(0, 0, total_lane_points, LANES_SPEED_LIMIT_SHAPE[2]) =
    lane_segments_matrix.block(0, SPEED_LIMIT, total_lane_points, LANES_SPEED_LIMIT_SHAPE[2]);
  lane_speed_matrix.transposeInPlace();
  return {lane_speed_matrix.data(), lane_speed_matrix.data() + lane_speed_matrix.size()};
}

std::vector<float> get_route_segments(
  const Eigen::MatrixXf & map_lane_segments_matrix, LaneletRoute::ConstSharedPtr route_ptr_,
  const RowLaneIDMaps & row_id_mapping)
{
  const auto total_route_points = ROUTE_LANES_SHAPE[1] * POINTS_PER_SEGMENT;
  Eigen::MatrixXf full_route_segment_matrix(total_route_points, SEGMENT_POINT_DIM);
  full_route_segment_matrix.setZero();
  long route_segment_rows = 0;

  for (const auto & route_segment : route_ptr_->segments) {
    auto route_segment_row_itr =
      row_id_mapping.lane_id_to_matrix_row.find(route_segment.preferred_primitive.id);
    if (route_segment_row_itr == row_id_mapping.lane_id_to_matrix_row.end()) {
      continue;
    }
    full_route_segment_matrix.block(route_segment_rows, 0, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM) =
      map_lane_segments_matrix.block(
        route_segment_row_itr->second, 0, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM);

    route_segment_rows += POINTS_PER_SEGMENT;
  }

  full_route_segment_matrix.transposeInPlace();
  return {
    full_route_segment_matrix.data(),
    full_route_segment_matrix.data() + full_route_segment_matrix.size()};
}

}  // namespace autoware::diffusion_planner::preprocess
