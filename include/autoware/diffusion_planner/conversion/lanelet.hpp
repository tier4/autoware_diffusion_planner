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

#ifndef AUTOWARE__DIFFUSION_PLANNER__CONVERSION__LANELET_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__CONVERSION__LANELET_HPP_

#include "autoware/diffusion_planner/polyline.hpp"

#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <geometry_msgs/msg/detail/point__struct.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/CompoundPolygon.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/utility/Optional.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autoware::diffusion_planner
{
using TrafficSignalElement = autoware_perception_msgs::msg::TrafficLightElement;
using TrafficSignal = autoware_perception_msgs::msg::TrafficLightGroup;
using TrafficSignalArray = autoware_perception_msgs::msg::TrafficLightGroupArray;
using TrafficLightIdMap = std::unordered_map<lanelet::Id, TrafficSignal>;
using autoware_perception_msgs::msg::TrafficLightElement;

constexpr long LANE_POINTS = 20;

/**
 * @brief Insert lane points into the container from the end of it.
 *
 * @param points Sequence of points to be inserted.
 * @param container Points container.
 */
inline void insertLanePoints(
  const std::vector<LanePoint> & points, std::vector<LanePoint> & container)
{
  container.reserve(container.size() * 2);
  container.insert(container.end(), points.begin(), points.end());
}

inline lanelet::Optional<std::string> toTypeName(const lanelet::ConstLanelet & lanelet)
{
  return lanelet.hasAttribute("type") ? lanelet.attribute("type").as<std::string>()
                                      : lanelet::Optional<std::string>();
}

inline lanelet::Optional<std::string> toTypeName(const lanelet::ConstLineString3d & linestring)
{
  return linestring.hasAttribute("type") ? linestring.attribute("type").as<std::string>()
                                         : lanelet::Optional<std::string>();
}

/**
 * @brief Extract the subtype name from a lanelet.
 *
 * @param lanelet Lanelet instance.
 * @return std::optional<string>
 */
inline lanelet::Optional<std::string> toSubtypeName(const lanelet::ConstLanelet & lanelet) noexcept
{
  return lanelet.hasAttribute("subtype") ? lanelet.attribute("subtype").as<std::string>()
                                         : lanelet::Optional<std::string>();
}

/**
 * @brief Extract the subtype name from a 3D linestring.
 *
 * @param linestring 3D linestring instance.
 * @return lanelet::Optional<std::string>
 */
inline lanelet::Optional<std::string> toSubtypeName(
  const lanelet::ConstLineString3d & linestring) noexcept
{
  return linestring.hasAttribute("subtype") ? linestring.attribute("subtype").as<std::string>()
                                            : lanelet::Optional<std::string>();
}

/**
 * @brief Check if the specified lanelet is the turnable intersection.
 *
 * @param lanelet Lanelet instance.
 * @return true if the lanelet has the attribute named turn_direction.
 */
inline bool isTurnableIntersection(const lanelet::ConstLanelet & lanelet) noexcept
{
  return lanelet.hasAttribute("turn_direction");
}

/**
 * @brief Check if the specified lanelet subtype is kind of lane.
 *
 * @param subtype
 * @return True if the lanelet subtype is the one of the (road, highway, road_shoulder,
 * pedestrian_lane, bicycle_lane, walkway).
 */
inline bool isLaneLike(const lanelet::Optional<std::string> & subtype)
{
  if (!subtype) {
    return false;
  }
  const auto & subtype_str = subtype.value();
  return (
    subtype_str == "road" || subtype_str == "highway" || subtype_str == "road_shoulder" ||
    subtype_str == "bicycle_lane");
  // subtype_str == "pedestrian_lane" || subtype_str == "bicycle_lane" || subtype_str == "walkway"
}

/**
 * @brief Check if the specified lanelet subtype is kind of the roadway.
 *
 * @param subtype Subtype of the corresponding lanelet.
 * @return True if the subtype is the one of the (road, highway, road_shoulder).
 */
inline bool isRoadwayLike(const lanelet::Optional<std::string> & subtype)
{
  if (!subtype) {
    return false;
  }
  const auto & subtype_str = subtype.value();
  return subtype_str == "road" || subtype_str == "highway" || subtype_str == "road_shoulder";
}

/**
 * @brief Check if the specified linestring is kind of the boundary.
 *
 * @param linestring 3D linestring.
 * @return True if the type is the one of the (line_thin, line_thick, road_boarder) and the subtype
 * is not virtual.
 */
inline bool isBoundaryLike(const lanelet::ConstLineString3d & linestring)
{
  const auto type = toTypeName(linestring);
  const auto subtype = toSubtypeName(linestring);
  if (!type || !subtype) {
    return false;
  }

  const auto & type_str = type.value();
  const auto & subtype_str = subtype.value();
  return (type_str == "line_thin" || type_str == "line_thick" || type_str == "road_boarder") &&
         subtype_str != "virtual";
}

/**
 * @brief Check if the specified linestring is the kind of crosswalk.
 *
 * @param subtype Subtype of the corresponding polygon.
 * @return True if the lanelet subtype is the one of the (crosswalk,).
 */
inline bool isCrosswalkLike(const lanelet::Optional<std::string> & subtype)
{
  if (!subtype) {
    return false;
  }

  const auto & subtype_str = subtype.value();
  return subtype_str == "crosswalk";
}

struct LaneSegment
{
  int64_t id;
  Polyline polyline;
  bool is_intersection{false};
  std::vector<BoundarySegment> left_boundaries;
  std::vector<BoundarySegment> right_boundaries;
  std::optional<float> speed_limit_mph{std::nullopt};
  uint8_t traffic_light;

  LaneSegment(
    int64_t id, Polyline polyline, bool is_intersection,
    const std::vector<BoundarySegment> & left_boundaries,
    const std::vector<BoundarySegment> & right_boundaries, std::optional<float> speed_limit_mph,
    const uint8_t traffic_light)
  : id(id),
    polyline(std::move(polyline)),
    is_intersection(is_intersection),
    left_boundaries(left_boundaries),
    right_boundaries(right_boundaries),
    speed_limit_mph(speed_limit_mph),
    traffic_light(traffic_light)
  {
  }
};

/**
 * @brief A class to convert lanelet map to polyline.
 */
class LaneletConverter
{
public:
  /**
   * @brief Construct a new Lanelet Converter object
   *
   * @param lanelet_map_ptr Pointer of loaded lanelet map.
   * @param max_num_polyline The max number of polylines to be contained in the tensor. If the total
   * number of polylines are less than this value, zero-filled polylines will be padded.
   * @param max_num_point The max number of points to be contained in a single polyline.
   * @param point_break_distance Distance threshold to separate two polylines.
   */
  explicit LaneletConverter(
    lanelet::LaneletMapConstPtr lanelet_map_ptr, size_t max_num_polyline, size_t max_num_point,
    float point_break_distance)
  : lanelet_map_ptr_(std::move(lanelet_map_ptr)),
    max_num_polyline_(max_num_polyline),
    max_num_point_(max_num_point),
    point_break_distance_(point_break_distance)
  {
  }

  /**
   * @brief Convert a lanelet map to the polyline data except of points whose distance from the
   * specified position is farther than the threshold.
   *
   * @param position Origin to check the distance from this.
   * @param distance_threshold Distance threshold
   * @return std::optional<PolylineData>
   */
  [[nodiscard]] std::optional<PolylineData> convert(
    const geometry_msgs::msg::Point & position, double distance_threshold) const;

  /**
   * @brief Convert a lanelet map to line segment data
   * @return std::vector<LaneSegment>
   */
  [[nodiscard]] std::vector<LaneSegment> convert_to_lane_segments() const;

  /**
   * @brief Convert lane segment data to matrix form
   * @return Eigen::MatrixXf
   */
  [[nodiscard]] Eigen::MatrixXf get_map_as_lane_segments(
    const std::vector<LaneSegment> & lane_segments);

  [[nodiscard]] Eigen::MatrixXf process_segment_to_matrix(
    const LaneSegment & segment, float center_x, float center_y, float mask_range) const;

  [[nodiscard]] Eigen::MatrixXf process_segments_to_matrix(
    const std::vector<LaneSegment> & segments, float center_x, float center_y,
    float mask_range) const;

private:
  /**
   * @brief Convert a linestring to the set of polylines.
   *
   * @param linestring Linestring instance.
   * @param position Origin to check the distance from this.
   * @param distance_threshold Distance threshold from the specified position.
   * @return std::vector<LanePoint>
   */
  [[nodiscard]] std::vector<LanePoint> fromLinestring(
    const lanelet::ConstLineString3d & linestring, const geometry_msgs::msg::Point & position,
    double distance_threshold) const noexcept;

  [[nodiscard]] std::vector<LanePoint> fromLinestring(
    const lanelet::ConstLineString3d & linestring) const noexcept;

  /**
   * @brief Convert a polygon to the set of polylines.
   *
   * @param polygon Polygon instance.
   * @param position Origin to check the distance from this.
   * @param distance_threshold Distance threshold from the specified position.
   * @return std::vector<LanePoint>
   */
  [[nodiscard]] std::vector<LanePoint> fromPolygon(
    const lanelet::CompoundPolygon3d & polygon, const geometry_msgs::msg::Point & position,
    double distance_threshold) const noexcept;

  [[nodiscard]] std::vector<LanePoint> fromPolygon(
    const lanelet::CompoundPolygon3d & polygon) const noexcept;

  lanelet::LaneletMapConstPtr lanelet_map_ptr_;  //!< Pointer of lanelet map.
  size_t max_num_polyline_;                      //!< The max number of polylines.
  size_t max_num_point_;                         //!< The max number of points.
  float point_break_distance_;                   //!< Distance threshold to separate two polylines.
};
struct RowWithDistance
{
  long index;
  float distance_squared;
};
// Function to compute squared distances of each matrix of lane segments
inline void compute_distances(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  std::vector<RowWithDistance> & distances)
{
  const auto n = input_matrix.rows();
  distances.clear();
  distances.reserve(n);
  for (long i = 0; i < n; i += LANE_POINTS) {
    // Directly access input matrix as raw memory
    float x = input_matrix(i, 0);
    float y = input_matrix(i, 1);
    Eigen::Vector4f p(x, y, 0.0f, 1.0f);
    Eigen::Vector4f p_transformed = transform_matrix * p;
    float distance_squared = p_transformed.head<2>().squaredNorm();
    distances.push_back({i, distance_squared});
  }
}

inline void sort_indices_by_distance(std::vector<RowWithDistance> & distances)
{
  std::sort(distances.begin(), distances.end(), [&](auto & a, auto & b) {
    return a.distance_squared < b.distance_squared;
  });
}

inline void transform_selected_cols(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation = true)
{
  Eigen::Matrix<float, 4, LANE_POINTS> xy_block = Eigen::Matrix<float, 4, LANE_POINTS>::Zero();
  xy_block.block<2, LANE_POINTS>(0, 0) =
    output_matrix.block<2, LANE_POINTS>(row_idx, column_idx * LANE_POINTS);
  xy_block.row(3) = do_translation ? Eigen::Matrix<float, 1, LANE_POINTS>::Ones()
                                   : Eigen::Matrix<float, 1, LANE_POINTS>::Zero();

  Eigen::Matrix<float, 4, LANE_POINTS> transformed_block = transform_matrix * xy_block;
  output_matrix.block<2, LANE_POINTS>(row_idx, column_idx * LANE_POINTS) =
    transformed_block.block<2, LANE_POINTS>(0, 0);
}

inline Eigen::MatrixXf transform_xy_points(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix,
  const std::vector<RowWithDistance> & distances, long m)
{
  constexpr int kCols = 14;

  const long n_total_segments = static_cast<int>(input_matrix.rows() / LANE_POINTS);
  const long num_segments = std::min(m, n_total_segments);
  const long num_rows = num_segments * LANE_POINTS;

  if (input_matrix.cols() < kCols) {
    throw std::invalid_argument("input_matrix must have at least 14 columns.");
  }

  Eigen::MatrixXf output_matrix(num_rows, kCols);
  output_matrix.setZero();
  output_matrix.transposeInPlace();  // helps to simplify the code below

  long col_counter = 0;
  for (auto itr = distances.begin(), end = distances.begin() + num_segments; itr != end; ++itr) {
    // get the 20 rows corresponding to the segment
    const auto row_idx = itr->index;

    output_matrix.block<kCols, LANE_POINTS>(0, col_counter * LANE_POINTS) =
      input_matrix.block<LANE_POINTS, kCols>(row_idx, 0).transpose();

    // transform the x and y coordinates
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 0);
    // the dx and dy coordinates do not require translation
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 2, false);
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 4);
    transform_selected_cols(transform_matrix, output_matrix, col_counter, 6);
    ++col_counter;
  }
  return output_matrix.transpose();
}

inline Eigen::MatrixXf transform_and_select_rows(
  const Eigen::MatrixXf & input_matrix, const Eigen::Matrix4f & transform_matrix, int m)
{
  const auto n = input_matrix.rows();
  if (n == 0 || input_matrix.cols() != 14 || m <= 0) {
    throw std::invalid_argument(
      "Input matrix must have at least 14 columns and m must be greater than 0.");
    return {};
  }
  std::vector<RowWithDistance> distances;
  // Step 1: Compute distances
  compute_distances(input_matrix, transform_matrix, distances);
  // Step 2: Sort indices by distance
  sort_indices_by_distance(distances);
  // Step 3: Apply transformation to selected rows
  return transform_xy_points(input_matrix, transform_matrix, distances, m);
}

}  // namespace autoware::diffusion_planner

#endif  // AUTOWARE__DIFFUSION_PLANNER__CONVERSION__LANELET_HPP_
