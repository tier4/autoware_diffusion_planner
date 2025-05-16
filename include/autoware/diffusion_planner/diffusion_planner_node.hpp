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

#ifndef AUTOWARE__DIFFUSION_PLANNER__DIFFUSION_PLANNER_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__DIFFUSION_PLANNER_HPP_

#include "autoware/diffusion_planner/conversion/agent.hpp"
#include "autoware/diffusion_planner/conversion/lanelet.hpp"
#include "autoware/diffusion_planner/dimensions.hpp"
#include "autoware/diffusion_planner/preprocessing/lane_segments.hpp"
#include "autoware/diffusion_planner/preprocessing/traffic_signals.hpp"
#include "autoware/diffusion_planner/utils/arg_reader.hpp"
#include "autoware_utils/ros/polling_subscriber.hpp"
#include "autoware_utils/system/time_keeper.hpp"
#include "builtin_interfaces/msg/duration.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "rclcpp/rclcpp.hpp"

#include <Eigen/Dense>
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_utils/ros/update_param.hpp>
#include <autoware_vehicle_info_utils/vehicle_info_utils.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/timer.hpp>

#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_signal.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>

#include <Eigen/src/Core/Matrix.h>
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRules.h>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::diffusion_planner
{
using autoware::diffusion_planner::AgentData;
using autoware_map_msgs::msg::LaneletMapBin;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_perception_msgs::msg::TrackedObjects;
using autoware_planning_msgs::msg::LaneletRoute;
using autoware_planning_msgs::msg::Trajectory;
using autoware_planning_msgs::msg::TrajectoryPoint;
using geometry_msgs::msg::AccelWithCovarianceStamped;
using nav_msgs::msg::Odometry;
using HADMapBin = autoware_map_msgs::msg::LaneletMapBin;
using InputDataMap = std::unordered_map<std::string, std::vector<float>>;
using builtin_interfaces::msg::Duration;
using builtin_interfaces::msg::Time;
using geometry_msgs::msg::Point;
using preprocess::RowLaneIDMaps;
using preprocess::TrafficSignalStamped;
using rcl_interfaces::msg::SetParametersResult;
using std_msgs::msg::ColorRGBA;
using utils::NormalizationMap;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
struct DiffusionPlannerParams
{
  std::string model_path;
  std::string args_path;
  double planning_frequency_hz;
};
struct DiffusionPlannerDebugParams
{
  bool publish_debug_route{false};
  bool publish_debug_map{false};
};

class DiffusionPlanner : public rclcpp::Node
{
public:
  explicit DiffusionPlanner(const rclcpp::NodeOptions & options);
  void set_up_params();
  void on_timer();
  void on_map(const HADMapBin::ConstSharedPtr map_msg);
  void load_model(const std::string & model_path);
  SetParametersResult on_parameter(const std::vector<rclcpp::Parameter> & parameters);

  // preprocessing
  std::pair<Eigen::Matrix4f, Eigen::Matrix4f> transforms_;
  AgentData get_ego_centric_agent_data(
    const TrackedObjects & objects, const Eigen::Matrix4f & map_to_ego_transform);

  InputDataMap create_input_data();

  // postprocessing
  Trajectory create_trajectory(
    std::vector<Ort::Value> & predictions, Eigen::Matrix4f & transform_ego_to_map);

  // current state
  Odometry ego_kinematic_state_;

  // onnxruntime
  OrtCUDAProviderOptions cuda_options_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Model input data
  std::optional<AgentData> agent_data_{std::nullopt};

  // Node parameters
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
  DiffusionPlannerParams params_;
  DiffusionPlannerDebugParams debug_params_;
  NormalizationMap normalization_map_;

  // Lanelet map
  LaneletRoute::ConstSharedPtr route_ptr_;
  std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr_;
  std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr_;
  std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr_;
  std::unique_ptr<LaneletConverter> lanelet_converter_ptr_;
  std::vector<LaneSegment> lane_segments_;
  Eigen::MatrixXf map_lane_segments_matrix_;
  RowLaneIDMaps row_id_mapping_;
  std::map<lanelet::Id, TrafficSignalStamped> traffic_light_id_map_;
  bool is_map_loaded_{false};

  // Node elements
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<autoware_utils::ProcessingTimeDetail>::SharedPtr
    debug_processing_time_detail_pub_;
  rclcpp::Publisher<Trajectory>::SharedPtr pub_trajectory_{nullptr};
  rclcpp::Publisher<PredictedObjects>::SharedPtr pub_objects_{nullptr};
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_lane_marker_{nullptr};
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_route_marker_{nullptr};
  mutable std::shared_ptr<autoware_utils::TimeKeeper> time_keeper_{nullptr};
  autoware_utils::InterProcessPollingSubscriber<Odometry> sub_current_odometry_{
    this, "~/input/odometry"};
  autoware_utils::InterProcessPollingSubscriber<AccelWithCovarianceStamped>
    sub_current_acceleration_{this, "~/input/acceleration"};
  autoware_utils::InterProcessPollingSubscriber<TrackedObjects> sub_tracked_objects_{
    this, "~/input/tracked_objects"};
  autoware_utils::InterProcessPollingSubscriber<
    autoware_perception_msgs::msg::TrafficLightGroupArray>
    sub_traffic_signals_{this, "~/input/traffic_signals"};
  autoware_utils::InterProcessPollingSubscriber<
    LaneletRoute, autoware_utils::polling_policy::Newest>
    route_subscriber_{this, "~/input/route", rclcpp::QoS{1}.transient_local()};
  autoware_utils::InterProcessPollingSubscriber<
    LaneletMapBin, autoware_utils::polling_policy::Newest>
    vector_map_subscriber_{this, "~/input/vector_map", rclcpp::QoS{1}.transient_local()};
  rclcpp::Subscription<HADMapBin>::SharedPtr sub_map_;

  tf2_ros::Buffer tf_buffer_{get_clock()};
  tf2_ros::TransformListener tf_listener_{tf_buffer_};
};

std::vector<float> load_tensor(const std::string & filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  if (size % sizeof(float) != 0)
    throw std::runtime_error("File size is not aligned with float size");

  file.seekg(0, std::ios::beg);
  std::vector<float> buffer(size / sizeof(float));
  file.read(reinterpret_cast<char *>(buffer.data()), size);
  return buffer;
}
}  // namespace autoware::diffusion_planner
#endif  // AUTOWARE__DIFFUSION_PLANNER__DIFFUSION_PLANNER_HPP_
