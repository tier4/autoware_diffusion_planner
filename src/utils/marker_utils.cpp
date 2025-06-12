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

#include "autoware/diffusion_planner/utils/marker_utils.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

#include <builtin_interfaces/msg/duration.hpp>
#include <rclcpp/duration.hpp>

#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <geometry_msgs/msg/point.hpp>  // Include the header for Point

namespace autoware::diffusion_planner::utils
{
using geometry_msgs::msg::Point;

ColorRGBA get_traffic_light_color(float g, float y, float r, const ColorRGBA & original_color)
{
  ColorRGBA color;
  color.r = 0.0;
  color.g = 0.0;
  color.b = 0.0;
  color.a = 0.8;
  if (static_cast<bool>(g)) {
    color.g = 1.0;
    return color;
  }
  if (static_cast<bool>(y)) {
    color.g = 1.0;
    color.r = 1.0;
    return color;
  }

  if (static_cast<bool>(r)) {
    color.r = 1.0;
    return color;
  }
  return original_color;
};

MarkerArray create_lane_marker(
  const Eigen::Matrix4f & transform_ego_to_map, const std::vector<float> & lane_vector,
  const std::vector<long> & shape, const Time & stamp, const rclcpp::Duration & lifetime,
  const std::array<float, 4> colors, const std::string & frame_id,
  const bool set_traffic_light_color)
{
  MarkerArray marker_array;
  const long P = shape[2];
  const long D = shape[3];
  long segment_count = 0;

  ColorRGBA color;
  color.r = colors[0];
  color.g = colors[1];
  color.b = colors[2];
  color.a = colors[3];

  ColorRGBA color_bounds;
  color_bounds.r = 0.9;
  color_bounds.g = 0.65;
  color_bounds.b = 0.0;
  color_bounds.a = 0.8;

  for (size_t l = 0; l < lane_vector.size() / (P * D); ++l) {
    Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame_id;
    marker.ns = "lane";
    marker.id = static_cast<int>(l);
    marker.type = Marker::LINE_STRIP;
    marker.action = Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.3;
    marker.color = color;
    marker.lifetime = lifetime;

    Marker marker_lb;
    marker_lb.header.stamp = stamp;
    marker_lb.header.frame_id = frame_id;
    marker_lb.ns = "lane_lb";
    marker_lb.id = static_cast<int>(l);
    marker_lb.type = Marker::LINE_STRIP;
    marker_lb.action = Marker::ADD;
    marker_lb.pose.orientation.w = 1.0;
    marker_lb.scale.x = 0.3;
    marker_lb.color = color_bounds;
    marker_lb.lifetime = lifetime;

    Marker marker_rb;
    marker_rb.header.stamp = stamp;
    marker_rb.header.frame_id = frame_id;
    marker_rb.ns = "lane_rb";
    marker_rb.id = static_cast<int>(l);
    marker_rb.type = Marker::LINE_STRIP;
    marker_rb.action = Marker::ADD;
    marker_rb.pose.orientation.w = 1.0;
    marker_rb.scale.x = 0.3;
    marker_rb.color = color_bounds;
    marker_rb.lifetime = lifetime;

    Marker marker_sphere;
    marker_sphere.header.stamp = stamp;
    marker_sphere.header.frame_id = frame_id;
    marker_sphere.ns = "sphere";
    marker_sphere.id = static_cast<int>(l);
    marker_sphere.type = Marker::SPHERE_LIST;
    marker_sphere.action = Marker::ADD;
    marker_sphere.pose.orientation.w = 1.0;
    marker_sphere.scale.x = 0.5;
    marker_sphere.scale.y = 0.5;
    marker_sphere.scale.z = 0.5;
    marker_sphere.lifetime = lifetime;

    ColorRGBA color_sphere;
    color_sphere.r = segment_count % 2 == 0 ? 0.1 : 0.9;
    color_sphere.g = segment_count % 2 == 0 ? 0.9 : 0.1;
    color_sphere.b = 0.9;
    color_sphere.a = 0.8;
    marker_sphere.color = color_sphere;

    if (set_traffic_light_color) {
      auto g = lane_vector[P * D * l + 0 * D + TRAFFIC_LIGHT_GREEN];
      auto y = lane_vector[P * D * l + 0 * D + TRAFFIC_LIGHT_YELLOW];
      auto r = lane_vector[P * D * l + 0 * D + TRAFFIC_LIGHT_RED];
      marker.color = get_traffic_light_color(g, y, r, color);
    }

    // Check if the centerline is all zeros
    float total_norm = 0.f;
    for (long p = 0; p < P; ++p) {
      auto x = lane_vector[P * D * l + p * D + X];
      auto y = lane_vector[P * D * l + p * D + Y];
      auto lb_x = lane_vector[P * D * l + p * D + LB_X] + x;
      auto lb_y = lane_vector[P * D * l + p * D + LB_Y] + y;
      auto rb_x = lane_vector[P * D * l + p * D + RB_X] + x;
      auto rb_y = lane_vector[P * D * l + p * D + RB_Y] + y;

      Eigen::Matrix<float, 4, 3> points;
      points << x, lb_x, rb_x, y, lb_y, rb_y, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f;

      // Apply transform
      Eigen::Matrix<float, 4, 3> transformed = transform_ego_to_map * points;

      // Assign back transformed values
      x = transformed(0, 0);
      y = transformed(1, 0);
      lb_x = transformed(0, 1);
      lb_y = transformed(1, 1);
      rb_x = transformed(0, 2);
      rb_y = transformed(1, 2);

      float z = transformed(2, 0) + 0.1f;
      total_norm += std::sqrt(x * x + y * y);

      Point pt;
      pt.x = x;
      pt.y = y;
      pt.z = z;
      marker.points.push_back(pt);

      Point lb_pt;
      lb_pt.x = lb_x;
      lb_pt.y = lb_y;
      lb_pt.z = z;
      marker_lb.points.push_back(lb_pt);

      Point rb_pt;
      rb_pt.x = rb_x;
      rb_pt.y = rb_y;
      rb_pt.z = z;
      marker_rb.points.push_back(rb_pt);

      Point pt_sphere;
      pt_sphere.x = x;
      pt_sphere.y = y;
      pt_sphere.z = z + 0.1f;
      marker_sphere.points.push_back(pt_sphere);
    }
    if (total_norm < 1e-2) {
      continue;
    }
    ++segment_count;

    if (!marker_sphere.points.empty()) {
      marker_array.markers.push_back(marker_sphere);
    }
    if (!marker.points.empty()) {
      marker_array.markers.push_back(marker);
    }
    if (!marker_lb.points.empty()) {
      marker_array.markers.push_back(marker_lb);
    }
    if (!marker_rb.points.empty()) {
      marker_array.markers.push_back(marker_rb);
    }
  }
  return marker_array;
}

}  // namespace autoware::diffusion_planner::utils
