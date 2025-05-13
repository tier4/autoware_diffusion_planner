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

#include "autoware/diffusion_planner/diffusion_planner_node.hpp"

#include "autoware/diffusion_planner/conversion/agent.hpp"
#include "autoware/diffusion_planner/conversion/ego.hpp"
#include "autoware/diffusion_planner/preprocessing/lane_segments.hpp"
#include "autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp"
#include "autoware/diffusion_planner/utils/utils.hpp"
#include "onnxruntime_cxx_api.h"

#include <autoware_utils/math/normalization.hpp>

#include <Eigen/src/Core/Matrix.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace autoware::diffusion_planner
{
DiffusionPlanner::DiffusionPlanner(const rclcpp::NodeOptions & options)
: Node("diffusion_planner", options), session_(nullptr)  // Initialize session_ with a default value
{
  // Initialize the node
  rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("diffusion_planner", options);
  pub_trajectory_ = node->create_publisher<Trajectory>("~/output/trajectory", 1);
  pub_route_marker_ = node->create_publisher<MarkerArray>("~/debug/route_marker", 10);
  pub_lane_marker_ = node->create_publisher<MarkerArray>("~/debug/lane_marker", 10);
  debug_processing_time_detail_pub_ = node->create_publisher<autoware_utils::ProcessingTimeDetail>(
    "~/debug/processing_time_detail_ms", 1);
  time_keeper_ = std::make_shared<autoware_utils::TimeKeeper>(debug_processing_time_detail_pub_);

  set_up_params();
  // Load the model
  if (params_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Model path is not set");
    return;
  }

  normalization_map_ = utils::load_normalization_stats(params_.args_path);
  load_model(params_.model_path);

  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Rate(params_.planning_frequency_hz).period(),
    std::bind(&DiffusionPlanner::on_timer, this));

  sub_map_ = create_subscription<HADMapBin>(
    "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&DiffusionPlanner::on_map, this, std::placeholders::_1));

  // Parameter Callback
  set_param_res_ = add_on_set_parameters_callback(
    std::bind(&DiffusionPlanner::on_parameter, this, std::placeholders::_1));
}

void DiffusionPlanner::set_up_params()
{
  params_.model_path = this->declare_parameter<std::string>("onnx_model_path", "");
  params_.args_path = this->declare_parameter<std::string>("args_path", "");
  params_.planning_frequency_hz = this->declare_parameter<double>("planning_frequency_hz", 10.0);

  debug_params_.publish_debug_map = this->declare_parameter<bool>("publish_debug_map", false);
  debug_params_.publish_debug_route = this->declare_parameter<bool>("publish_debug_route", false);
  RCLCPP_INFO(get_logger(), "Setting up parameters for Diffusion Planner");
}

SetParametersResult DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
  using autoware_utils::update_param;
  DiffusionPlannerDebugParams temp_debug_params = debug_params_;
  update_param<bool>(parameters, "publish_debug_map", temp_debug_params.publish_debug_map);
  update_param<bool>(parameters, "publish_debug_route", temp_debug_params.publish_debug_route);
  debug_params_ = temp_debug_params;

  SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void DiffusionPlanner::load_model(const std::string & model_path)
{
  env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DiffusionPlanner");
  session_options_.SetLogSeverityLevel(1);
  session_options_.AppendExecutionProvider_CUDA(cuda_options_);
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  session_ = Ort::Session(env_, model_path.c_str(), session_options_);
  RCLCPP_INFO(get_logger(), "Model loaded from %s", params_.model_path.c_str());
}

AgentData DiffusionPlanner::get_ego_centric_agent_data(
  const TrackedObjects & objects, const Eigen::Matrix4f & map_to_ego_transform)
{
  if (!agent_data_) {
    agent_data_ = AgentData(objects, NEIGHBOR_SHAPE[1], NEIGHBOR_SHAPE[2]);
  } else {
    agent_data_->update_histories(objects);
  }

  auto ego_centric_data = agent_data_.value();
  ego_centric_data.apply_transform(map_to_ego_transform);
  ego_centric_data.trim_to_k_closest_agents();
  return ego_centric_data;
}

MarkerArray DiffusionPlanner::create_lane_marker(
  const std::vector<float> & lane_vector, [[maybe_unused]] const std::vector<long> & shape,
  const Time & stamp, const std::array<float, 4> colors, const std::string & ns)
{
  MarkerArray marker_array;
  const long P = shape[2];
  const long D = shape[3];
  long sphere_count = 0;

  ColorRGBA color;
  color.r = colors[0];
  color.g = colors[1];
  color.b = colors[2];
  color.a = colors[3];

  Duration lifetime;
  lifetime.sec = 0;
  lifetime.nanosec = 1e8;

  for (size_t l = 0; l < lane_vector.size() / (P * D); ++l) {
    // Check if the centerline is all zeros
    Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = ns;
    marker.ns = "lane";
    marker.id = static_cast<int>(l);
    marker.type = Marker::LINE_STRIP;
    marker.action = Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.3;
    marker.color = color;
    marker.lifetime = lifetime;

    Marker marker_sphere;
    marker_sphere.header.stamp = stamp;
    marker_sphere.header.frame_id = ns;
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
    color_sphere.r = sphere_count % 2 == 0 ? 0.1 : 0.9;
    color_sphere.g = sphere_count % 2 == 0 ? 0.9 : 0.1;
    color_sphere.b = 0.0;
    color_sphere.a = 0.8;
    marker_sphere.color = color_sphere;

    for (long p = 0; p < P; ++p) {
      auto x = lane_vector[P * D * l + p * D + 0];
      auto y = lane_vector[P * D * l + p * D + 1];
      float z = 0.5f;
      float norm = std::sqrt(x * x + y * y);
      if (norm < 1e-2) continue;

      Point pt;
      pt.x = x;
      pt.y = y;
      pt.z = z;
      marker.points.push_back(pt);

      Point pt_sphere;
      pt_sphere.x = x;
      pt_sphere.y = y;
      pt_sphere.z = sphere_count % 2 == 0 ? 0.5 : 1.0;
      marker_sphere.points.push_back(pt_sphere);
    }
    ++sphere_count;

    if (!marker_sphere.points.empty()) {
      marker_array.markers.push_back(marker_sphere);
    }
    if (!marker.points.empty()) {
      marker_array.markers.push_back(marker);
    }
  }

  return marker_array;
}

InputDataMap DiffusionPlanner::create_input_data()
{
  InputDataMap input_data_map;
  auto objects = sub_tracked_objects_.take_data();
  auto ego_kinematic_state = sub_current_odometry_.take_data();
  auto ego_acceleration = sub_current_acceleration_.take_data();
  auto temp_route = route_subscriber_.take_data();

  transforms_ = utils::get_transform_matrix(*ego_kinematic_state);
  route_ptr_ = (!route_ptr_ || temp_route) ? temp_route : route_ptr_;

  if (!objects || !ego_kinematic_state || !ego_acceleration || !route_ptr_) {
    RCLCPP_WARN(get_logger(), "No tracked objects or ego kinematic state or route data received");
    return {};
  }
  // Ego state
  // TODO(Daniel): use vehicle_info_utils
  EgoState ego_state(*ego_kinematic_state, *ego_acceleration, 5.0);
  input_data_map["ego_current_state"] = ego_state.as_array();
  // Agent data on ego reference frame
  auto map_to_ego_transform = transforms_.second;
  input_data_map["neighbor_agents_past"] =
    get_ego_centric_agent_data(*objects, map_to_ego_transform).as_vector();
  // Static objects
  // TODO(Daniel): add static objects
  input_data_map["static_objects"] = utils::create_float_data(STATIC_OBJECTS_SHAPE, 0.0f);

  // map data on ego reference frame
  const auto center_x = static_cast<float>(ego_kinematic_state->pose.pose.position.x);
  const auto center_y = static_cast<float>(ego_kinematic_state->pose.pose.position.y);
  Eigen::MatrixXf ego_centric_lane_segments = preprocess::transform_and_select_rows(
    map_lane_segments_matrix_, map_to_ego_transform, center_x, center_y, LANES_SHAPE[1]);
  input_data_map["lanes"] = preprocess::extract_lane_tensor_data(ego_centric_lane_segments);
  input_data_map["lanes_speed_limit"] =
    preprocess::extract_lane_speed_tensor_data(ego_centric_lane_segments);

  // route data on ego reference frame
  input_data_map["route_lanes"] = preprocess::get_route_segments(
    map_lane_segments_matrix_, map_to_ego_transform, route_ptr_, segment_row_indices_, center_x,
    center_y);
  return input_data_map;
}

Trajectory DiffusionPlanner::create_trajectory(
  std::vector<Ort::Value> & predictions, Eigen::Matrix4f & transform_ego_to_map)
{
  Trajectory trajectory;
  trajectory.header.stamp = this->now();
  trajectory.header.frame_id = "map";
  // one batch of predictions
  // TODO(Daniel): add batch support
  auto data = predictions[0].GetTensorMutableData<float>();
  const auto prediction_shape = predictions[0].GetTensorTypeAndShapeInfo().GetShape();
  const auto num_of_dimensions = prediction_shape.size();

  // copy relevant part of data to Eigen matrix
  auto rows = prediction_shape[num_of_dimensions - 2];
  auto cols = prediction_shape[num_of_dimensions - 1];

  Eigen::MatrixXf prediction_matrix(rows, cols);

  // Fill matrix row-wise from data using Eigen::Map
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped_data(
    data, rows, cols);

  // Copy only the relevant part
  prediction_matrix = mapped_data;  // Copies first rows*cols elements row-wise
  prediction_matrix.transposeInPlace();
  transform_output_matrix(transform_ego_to_map, prediction_matrix, 0, 0, true);
  transform_output_matrix(transform_ego_to_map, prediction_matrix, 0, 2, false);
  prediction_matrix.transposeInPlace();

  constexpr float dt = 0.1f;
  float prev_x = 0.f;
  float prev_y = 0.f;
  for (long row = 0; row < prediction_matrix.rows(); ++row) {
    TrajectoryPoint p;
    p.pose.position.x = prediction_matrix(row, 0);
    p.pose.position.y = prediction_matrix(row, 1);
    p.pose.position.z = 0.0;
    auto yaw = std::atan2(prediction_matrix(row, 3), prediction_matrix(row, 2));
    yaw = static_cast<float>(autoware_utils::normalize_radian(yaw));
    p.pose.orientation = autoware_utils::create_quaternion_from_yaw(yaw);
    auto distance = std::hypotf(p.pose.position.x - prev_x, p.pose.position.y - prev_y);
    p.longitudinal_velocity_mps = distance / dt;

    prev_x = p.pose.position.x;
    prev_y = p.pose.position.y;
    trajectory.points.push_back(p);
  }
  return trajectory;
}

void DiffusionPlanner::on_timer()
{
  // Timer callback function
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);

  if (!is_map_loaded_) {
    RCLCPP_INFO(get_logger(), "Waiting for map data...");
    return;
  }

  // Prepare input data for the model
  auto input_data_map = create_input_data();
  if (input_data_map.empty()) {
    RCLCPP_WARN(get_logger(), "No input data available for inference");
    return;
  }

  if (debug_params_.publish_debug_route) {
    auto route_markers = create_lane_marker(
      input_data_map["route_lanes"], ROUTE_LANES_SHAPE, this->now(), {0.1, 0.8, 0.0, 0.8},
      "base_link");
    pub_route_marker_->publish(route_markers);
  }

  if (debug_params_.publish_debug_map) {
    auto lane_markers = create_lane_marker(
      input_data_map["lanes"], LANES_SHAPE, this->now(), {0.1, 0.1, 0.7, 0.8}, "base_link");
    pub_lane_marker_->publish(lane_markers);
  }

  // normalization of data
  preprocess::normalize_input_data(input_data_map, normalization_map_);

  auto ego_current_state = input_data_map["ego_current_state"];
  auto neighbor_agents_past = input_data_map["neighbor_agents_past"];
  auto static_objects = input_data_map["static_objects"];
  auto lanes = input_data_map["lanes"];
  auto lanes_speed_limit = input_data_map["lanes_speed_limit"];
  auto route_lanes = input_data_map["route_lanes"];

  // Allocate raw memory for bool array
  size_t lane_speed_tensor_num_elements = std::accumulate(
    LANES_SPEED_LIMIT_SHAPE.begin(), LANES_SPEED_LIMIT_SHAPE.end(), 1, std::multiplies<int64_t>());
  auto raw_speed_bool_array =
    std::shared_ptr<bool>(new bool[lane_speed_tensor_num_elements], std::default_delete<bool[]>());

  for (size_t i = 0; i < lane_speed_tensor_num_elements; ++i) {
    raw_speed_bool_array.get()[i] = (lanes_speed_limit[i] > std::numeric_limits<float>::epsilon());
  }

  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session_, mem_info);
  auto ego_current_state_tensor = Ort::Value::CreateTensor<float>(
    mem_info, ego_current_state.data(), ego_current_state.size(), EGO_CURRENT_STATE_SHAPE.data(),
    EGO_CURRENT_STATE_SHAPE.size());
  auto neighbor_agents_past_tensor = Ort::Value::CreateTensor<float>(
    mem_info, neighbor_agents_past.data(), neighbor_agents_past.size(), NEIGHBOR_SHAPE.data(),
    NEIGHBOR_SHAPE.size());
  auto static_objects_tensor = Ort::Value::CreateTensor<float>(
    mem_info, static_objects.data(), static_objects.size(), STATIC_OBJECTS_SHAPE.data(),
    STATIC_OBJECTS_SHAPE.size());
  auto lanes_tensor = Ort::Value::CreateTensor<float>(
    mem_info, lanes.data(), lanes.size(), LANES_SHAPE.data(), LANES_SHAPE.size());
  auto lanes_speed_limit_tensor = Ort::Value::CreateTensor<float>(
    mem_info, lanes_speed_limit.data(), lanes_speed_limit.size(), LANES_SPEED_LIMIT_SHAPE.data(),
    LANES_SPEED_LIMIT_SHAPE.size());
  auto lane_has_speed_limit_tensor = Ort::Value::CreateTensor<bool>(
    mem_info, raw_speed_bool_array.get(), lane_speed_tensor_num_elements,
    LANE_HAS_SPEED_LIMIT_SHAPE.data(), LANE_HAS_SPEED_LIMIT_SHAPE.size());
  auto route_lanes_tensor = Ort::Value::CreateTensor<float>(
    mem_info, route_lanes.data(), route_lanes.size(), ROUTE_LANES_SHAPE.data(),
    ROUTE_LANES_SHAPE.size());

  Ort::Value input_tensors[] = {
    std::move(ego_current_state_tensor), std::move(neighbor_agents_past_tensor),
    std::move(static_objects_tensor),    std::move(lanes_tensor),
    std::move(lanes_speed_limit_tensor), std::move(lane_has_speed_limit_tensor),
    std::move(route_lanes_tensor)};

  const char * input_names[] = {
    "ego_current_state", "neighbor_agents_past",  "static_objects", "lanes",
    "lanes_speed_limit", "lanes_has_speed_limit", "route_lanes"};

  const char * output_names[] = {"output"};
  // run inference
  try {
    auto output =
      session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 7, output_names, 1);
    auto output_trajectory = create_trajectory(output, transforms_.first);
    pub_trajectory_->publish(output_trajectory);
  } catch (const Ort::Exception & e) {
    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
  }
}

void DiffusionPlanner::on_map(const HADMapBin::ConstSharedPtr map_msg)
{
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *map_msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);

  lanelet_converter_ptr_ = std::make_unique<LaneletConverter>(lanelet_map_ptr_, 100, 20, 100.0);
  lane_segments_ = lanelet_converter_ptr_->convert_to_lane_segments(POINTS_PER_SEGMENT);

  if (lane_segments_.empty()) {
    RCLCPP_WARN(get_logger(), "No lane segments found in the map");
    throw std::runtime_error("No lane segments found in the map");
  }

  map_lane_segments_matrix_ =
    preprocess::process_segments_to_matrix(lane_segments_, segment_row_indices_);

  is_map_loaded_ = true;
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
