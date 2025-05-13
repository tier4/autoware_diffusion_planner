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

namespace autoware::diffusion_planner::preprocess
{

void compute_distances(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  std::vector<RowWithDistance> & distances, float center_x, float center_y, float mask_range)
{
  const auto n = input_matrix.rows();
  distances.reserve(n);
  for (long i = 0; i < n; i += POINTS_PER_LANE_SEGMENT) {
    // Directly access input matrix as raw memory
    float x = input_matrix.block(i, 0, POINTS_PER_LANE_SEGMENT, 1).mean();
    float y = input_matrix.block(i, 1, POINTS_PER_LANE_SEGMENT, 1).mean();
    bool inside =
      (x > center_x - mask_range * 1.1 && x < center_x + mask_range * 1.1 &&
       y > center_y - mask_range * 1.1 && y < center_y + mask_range * 1.1);
    float x_first = input_matrix(i, 0);
    float y_first = input_matrix(i, 1);

    float x_last = input_matrix(i + POINTS_PER_LANE_SEGMENT - 1, 0);
    float y_last = input_matrix(i + POINTS_PER_LANE_SEGMENT - 1, 1);

    Eigen::Vector4f p_first(x_first, y_first, 0.0f, 1.0f);
    Eigen::Vector4f p_transformed_first = transform_matrix * p_first;
    float distance_squared_first = p_transformed_first.head<2>().squaredNorm();

    Eigen::Vector4f p_last(x_last, y_last, 0.0f, 1.0f);
    Eigen::Vector4f p_transformed_last = transform_matrix * p_last;
    float distance_squared_last = p_transformed_last.head<2>().squaredNorm();

    float distance_squared = std::min(distance_squared_last, distance_squared_first);
    distances.push_back({i, distance_squared, inside});
  }
}

void transform_selected_cols(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation)
{
  Eigen::Matrix<float, 4, POINTS_PER_LANE_SEGMENT> xy_block =
    Eigen::Matrix<float, 4, POINTS_PER_LANE_SEGMENT>::Zero();
  xy_block.block<2, POINTS_PER_LANE_SEGMENT>(0, 0) =
    output_matrix.block<2, POINTS_PER_LANE_SEGMENT>(row_idx, column_idx * POINTS_PER_LANE_SEGMENT);
  xy_block.row(3) = do_translation ? Eigen::Matrix<float, 1, POINTS_PER_LANE_SEGMENT>::Ones()
                                   : Eigen::Matrix<float, 1, POINTS_PER_LANE_SEGMENT>::Zero();

  Eigen::Matrix<float, 4, POINTS_PER_LANE_SEGMENT> transformed_block = transform_matrix * xy_block;
  output_matrix.block<2, POINTS_PER_LANE_SEGMENT>(row_idx, column_idx * POINTS_PER_LANE_SEGMENT) =
    transformed_block.block<2, POINTS_PER_LANE_SEGMENT>(0, 0);
}

Eigen::MatrixXf transform_xy_points(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  const std::vector<RowWithDistance> & distances, long m)
{
  const long n_total_segments = static_cast<long>(input_matrix.rows() / POINTS_PER_LANE_SEGMENT);
  const long num_segments = std::min(m, n_total_segments);

  if (input_matrix.cols() < FULL_MATRIX_COLS) {
    throw std::invalid_argument("input_matrix must have at least FULL_MATRIX_COLS columns.");
  }

  Eigen::MatrixXf output_matrix(m * POINTS_PER_LANE_SEGMENT, FULL_MATRIX_COLS);
  output_matrix.setZero();
  output_matrix.transposeInPlace();  // helps to simplify the code below

  long col_counter = 0;
  long added_segments = 0;
  for (auto distance : distances) {
    if (!distance.inside) {
      continue;
    }
    const auto row_idx = distance.index;
    // get POINTS_PER_LANE_SEGMENT rows corresponding to a single segment
    output_matrix.block<FULL_MATRIX_COLS, POINTS_PER_LANE_SEGMENT>(
      0, col_counter * POINTS_PER_LANE_SEGMENT) =
      input_matrix.block<POINTS_PER_LANE_SEGMENT, FULL_MATRIX_COLS>(row_idx, 0).transpose();

    // transform the x and y coordinates
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 0);
    // the dx and dy coordinates do not require translation
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 2, false);
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 4);
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 6);
    ++col_counter;
    ++added_segments;
    if (added_segments >= num_segments) {
      break;
    }
  }
  // subtract center from boundaries
  output_matrix.row(4) = output_matrix.row(4) - output_matrix.row(0);
  output_matrix.row(5) = output_matrix.row(5) - output_matrix.row(1);
  output_matrix.row(6) = output_matrix.row(6) - output_matrix.row(0);
  output_matrix.row(7) = output_matrix.row(7) - output_matrix.row(1);

  return output_matrix.transpose();
}

Eigen::MatrixXf transform_and_select_rows(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix, float center_x,
  float center_y, long m)
{
  const auto n = input_matrix.rows();
  if (n == 0 || input_matrix.cols() != FULL_MATRIX_COLS || m <= 0) {
    throw std::invalid_argument(
      "Input matrix must have at least FULL_MATRIX_COLS columns and m must be greater than 0.");
    return {};
  }
  std::vector<RowWithDistance> distances;
  // Step 1: Compute distances
  compute_distances(input_matrix, transform_matrix, distances, center_x, center_y, 100);
  // Step 2: Sort indices by distance
  sort_indices_by_distance(distances);
  // Step 3: Apply transformation to selected rows
  return transform_xy_points(input_matrix, transform_matrix, distances, m);
}

Eigen::MatrixXf process_segments_to_matrix(
  const std::vector<LaneSegment> & lane_segments, std::map<int64_t, long> & segment_row_indices,
  float center_x, float center_y, float mask_range)
{
  std::vector<Eigen::MatrixXf> all_segment_matrices;

  long total_rows = 0;

  for (const auto & segment : lane_segments) {
    Eigen::MatrixXf segment_matrix =
      process_segment_to_matrix(segment, center_x, center_y, mask_range);

    if (segment_matrix.rows() != POINTS_PER_LANE_SEGMENT) {
      throw std::runtime_error("Segment matrix rows not equal to 20");
    }
    total_rows += segment_matrix.rows();
    all_segment_matrices.push_back(segment_matrix);
  }

  // Now allocate the full matrix
  const long cols = all_segment_matrices.empty() ? 0 : all_segment_matrices[0].cols();
  Eigen::MatrixXf stacked_matrix(total_rows, cols);

  long current_row = 0;
  for (const auto & mat : all_segment_matrices) {
    stacked_matrix.middleRows(current_row, mat.rows()) = mat;
    auto id = static_cast<int64_t>(mat(0, 13));
    segment_row_indices.emplace(id, current_row);
    if (mat.rows() != POINTS_PER_LANE_SEGMENT) {
      throw std::runtime_error("(2)Segment matrix rows not equal to 20");
    }
    current_row += mat.rows();
  }
  return stacked_matrix;
}

Eigen::MatrixXf process_segment_to_matrix(
  const LaneSegment & segment, float center_x, float center_y, float mask_range)
{
  if (
    segment.polyline.is_empty() || segment.left_boundaries.empty() ||
    segment.right_boundaries.empty()) {
    return {};
  }
  const auto & centerlines = segment.polyline.waypoints();
  const auto & left_boundaries = segment.left_boundaries.front().waypoints();
  const auto & right_boundaries = segment.right_boundaries.front().waypoints();
  const auto & first_waypoint = segment.polyline.waypoints()[0];

  if (
    first_waypoint.x() < center_x - mask_range * 1.1f ||
    first_waypoint.x() > center_x + mask_range * 1.1f ||
    first_waypoint.y() < center_y - mask_range * 1.1f ||
    first_waypoint.y() > center_y + mask_range * 1.1f) {
    return {};
  }

  const size_t N = centerlines.size();
  if (left_boundaries.size() != N || right_boundaries.size() != N) {
    return {};
  }

  Eigen::MatrixXf segment_data(N, 14);  // 14 = 2 + 2 + 2 + 2 + 4 + 1 + 1

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
    case 4:
      traffic_light_vec[3] = 1.0f;
      break;  // WHITE
    default:
      traffic_light_vec[3] = 1.0f;
      break;  // UNKNOWN
  }

  // Build each row
  for (long i = 0; i < static_cast<long>(N); ++i) {
    segment_data(i, 0) = centerlines[i].x();
    segment_data(i, 1) = centerlines[i].y();
    segment_data(i, 2) =
      i < static_cast<long>(N) - 1 ? centerlines[i + 1].x() - centerlines[i].x() : 0.0f;
    segment_data(i, 3) =
      i < static_cast<long>(N) - 1 ? centerlines[i + 1].y() - centerlines[i].y() : 0.0f;
    segment_data(i, 4) = left_boundaries[i].x();
    segment_data(i, 5) = left_boundaries[i].y();
    segment_data(i, 6) = right_boundaries[i].x();
    segment_data(i, 7) = right_boundaries[i].y();
    segment_data.block<1, 4>(i, 8) = traffic_light_vec.transpose();
    segment_data(i, 12) = segment.speed_limit_mph.value_or(0.0f);
    segment_data(i, 13) = static_cast<float>(segment.id);
  }

  return segment_data;
}

}  // namespace autoware::diffusion_planner::preprocess
