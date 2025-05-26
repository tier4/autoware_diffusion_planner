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
#include "autoware/diffusion_planner/dimensions.hpp"
#include "autoware/diffusion_planner/postprocessing/postprocessing_utils.hpp"
#include "autoware/diffusion_planner/preprocessing/preprocessing_utils.hpp"
#include "autoware/diffusion_planner/utils/marker_utils.hpp"
#include "autoware/diffusion_planner/utils/utils.hpp"
#include "onnxruntime_cxx_api.h"

#include <autoware_lanelet2_extension/utility/query.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/logging.hpp>

#include <Eigen/src/Core/Matrix.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace autoware::diffusion_planner
{
DiffusionPlanner::DiffusionPlanner(const rclcpp::NodeOptions & options)
: Node("diffusion_planner", options),
  session_(nullptr),
  generator_uuid_(autoware_utils_uuid::generate_uuid())
{
  // Initialize the node
  pub_trajectory_ = this->create_publisher<Trajectory>("~/output/trajectory", 1);
  pub_trajectories_ = this->create_publisher<Trajectories>("~/output/trajectories", 1);
  pub_objects_ =
    this->create_publisher<PredictedObjects>("~/output/predicted_objects", rclcpp::QoS(1));
  pub_route_marker_ = this->create_publisher<MarkerArray>("~/debug/route_marker", 10);
  pub_lane_marker_ = this->create_publisher<MarkerArray>("~/debug/lane_marker", 10);
  debug_processing_time_detail_pub_ = this->create_publisher<autoware_utils::ProcessingTimeDetail>(
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
  // node params
  params_.model_path = this->declare_parameter<std::string>("onnx_model_path", "");
  params_.args_path = this->declare_parameter<std::string>("args_path", "");
  params_.planning_frequency_hz = this->declare_parameter<double>("planning_frequency_hz", 10.0);
  params_.predict_neighbor_trajectory =
    this->declare_parameter<bool>("predict_neighbor_trajectory", false);
  params_.update_traffic_light_group_info =
    this->declare_parameter<bool>("update_traffic_light_group_info", false);
  params_.traffic_light_group_msg_timeout_seconds =
    this->declare_parameter<double>("traffic_light_group_msg_timeout_seconds", 0.2);

  // debug params
  debug_params_.publish_debug_map =
    this->declare_parameter<bool>("debug_params.publish_debug_map", false);
  debug_params_.publish_debug_route =
    this->declare_parameter<bool>("debug_params.publish_debug_route", false);
  RCLCPP_INFO(get_logger(), "Setting up parameters for Diffusion Planner");
}

SetParametersResult DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
  using autoware_utils::update_param;
  {
    DiffusionPlannerParams temp_params = params_;
    update_param<bool>(
      parameters, "predict_neighbor_trajectory", temp_params.predict_neighbor_trajectory);
    params_ = temp_params;
    update_param<bool>(
      parameters, "update_traffic_light_group_info", temp_params.update_traffic_light_group_info);
    update_param<double>(
      parameters, "traffic_light_group_msg_timeout_seconds",
      temp_params.traffic_light_group_msg_timeout_seconds);
    params_ = temp_params;
  }

  {
    DiffusionPlannerDebugParams temp_debug_params = debug_params_;
    update_param<bool>(
      parameters, "debug_params.publish_debug_map", temp_debug_params.publish_debug_map);
    update_param<bool>(
      parameters, "debug_params.publish_debug_route", temp_debug_params.publish_debug_route);
    debug_params_ = temp_debug_params;
  }

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

  auto ego_centric_agent_data = agent_data_.value();
  ego_centric_agent_data.apply_transform(map_to_ego_transform);
  ego_centric_agent_data.trim_to_k_closest_agents();
  return ego_centric_agent_data;
}

InputDataMap DiffusionPlanner::create_input_data()
{
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);
  InputDataMap input_data_map;
  auto objects = sub_tracked_objects_.take_data();
  auto ego_kinematic_state = sub_current_odometry_.take_data();
  auto ego_acceleration = sub_current_acceleration_.take_data();
  auto traffic_signals = sub_traffic_signals_.take_data();
  auto temp_route_ptr = route_subscriber_.take_data();

  route_ptr_ = (!route_ptr_ || temp_route_ptr) ? temp_route_ptr : route_ptr_;

  if (!objects || !ego_kinematic_state || !ego_acceleration || !route_ptr_) {
    RCLCPP_WARN(get_logger(), "No tracked objects or ego kinematic state or route data received");
    return {};
  }

  route_handler_->setRoute(*route_ptr_);
  std::map<lanelet::Id, TrafficSignalStamped> traffic_light_id_map;
  if (params_.update_traffic_light_group_info) {
    const auto & traffic_light_msg_timeout_s = params_.traffic_light_group_msg_timeout_seconds;
    preprocess::process_traffic_signals(
      traffic_signals, traffic_light_id_map, this->now(), traffic_light_msg_timeout_s);
    if (!traffic_signals) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "no traffic signal received. traffic light info will not be updated/used");
    }
  }

  ego_kinematic_state_ = *ego_kinematic_state;
  transforms_ = utils::get_transform_matrix(*ego_kinematic_state);

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
  input_data_map["static_objects"] = utils::create_float_data(
    std::vector<int64_t>(STATIC_OBJECTS_SHAPE.begin(), STATIC_OBJECTS_SHAPE.end()), 0.0f);

  // map data on ego reference frame
  const auto center_x = static_cast<float>(ego_kinematic_state->pose.pose.position.x);
  const auto center_y = static_cast<float>(ego_kinematic_state->pose.pose.position.y);
  std::tuple<Eigen::MatrixXf, ColLaneIDMaps> matrix_mapping_tuple =
    preprocess::transform_and_select_rows(
      map_lane_segments_matrix_, map_to_ego_transform, col_id_mapping_, traffic_light_id_map,
      lanelet_map_ptr_, center_x, center_y, LANES_SHAPE[1]);
  const Eigen::MatrixXf & ego_centric_lane_segments = std::get<0>(matrix_mapping_tuple);
  input_data_map["lanes"] = preprocess::extract_lane_tensor_data(ego_centric_lane_segments);
  input_data_map["lanes_speed_limit"] =
    preprocess::extract_lane_speed_tensor_data(ego_centric_lane_segments);

  // route data on ego reference frame
  const auto & current_pose = ego_kinematic_state->pose.pose;
  lanelet::ConstLanelet current_preferred_lane;
  constexpr double backward_path_length{5.0};
  constexpr double forward_path_length{200.0};

  if (!route_handler_->getClosestPreferredLaneletWithinRoute(
        current_pose, &current_preferred_lane)) {
    auto clock{rclcpp::Clock{RCL_ROS_TIME}};
    RCLCPP_ERROR_STREAM_THROTTLE(
      rclcpp::get_logger("diffusion_planner").get_child("utils"), clock, 1000,
      "failed to find closest lanelet within route!!!");
    return {};
  }

  // For current_lanes with desired length
  auto current_lanes = route_handler_->getLaneletSequence(
    current_preferred_lane, backward_path_length, forward_path_length);

  input_data_map["route_lanes"] = preprocess::get_route_segments(
    map_lane_segments_matrix_, map_to_ego_transform, col_id_mapping_, traffic_light_id_map,
    lanelet_map_ptr_, current_lanes);
  return input_data_map;
}

void DiffusionPlanner::publish_debug_markers(InputDataMap & input_data_map) const
{
  if (debug_params_.publish_debug_route) {
    auto lifetime = rclcpp::Duration::from_seconds(0.1);
    auto route_markers = utils::create_lane_marker(
      input_data_map["route_lanes"],
      std::vector<long>(ROUTE_LANES_SHAPE.begin(), ROUTE_LANES_SHAPE.end()), this->now(), lifetime,
      {0.8, 0.8, 0.8, 0.8}, "base_link", true);
    pub_route_marker_->publish(route_markers);
  }

  if (debug_params_.publish_debug_map) {
    auto lifetime = rclcpp::Duration::from_seconds(0.1);
    auto lane_markers = utils::create_lane_marker(
      input_data_map["lanes"], std::vector<long>(LANES_SHAPE.begin(), LANES_SHAPE.end()),
      this->now(), lifetime, {0.1, 0.1, 0.7, 0.8}, "base_link", true);
    pub_lane_marker_->publish(lane_markers);
  }
}

void DiffusionPlanner::publish_predictions(Ort::Value & predictions) const
{
  constexpr long batch_idx = 0;
  constexpr long ego_agent_idx = 0;
  auto output_trajectory = postprocessing::create_trajectory(
    predictions, this->now(), transforms_.first, batch_idx, ego_agent_idx);
  pub_trajectory_->publish(output_trajectory);

  auto ego_trajectory_as_new_msg =
    postprocessing::to_trajectories_msg(output_trajectory, generator_uuid_, "DiffusionPlanner");
  pub_trajectories_->publish(ego_trajectory_as_new_msg);

  // Other agents prediction
  if (params_.predict_neighbor_trajectory && agent_data_.has_value()) {
    auto reduced_agent_data = agent_data_.value();
    reduced_agent_data.trim_to_k_closest_agents(ego_kinematic_state_.pose.pose.position);
    auto predicted_objects = postprocessing::create_predicted_objects(
      predictions, reduced_agent_data, this->now(), transforms_.first);
    pub_objects_->publish(predicted_objects);
  }
}

std::optional<std::vector<Ort::Value>> DiffusionPlanner::do_inference(InputDataMap & input_data_map)
{
  auto & ego_current_state = input_data_map["ego_current_state"];
  auto & neighbor_agents_past = input_data_map["neighbor_agents_past"];
  auto & static_objects = input_data_map["static_objects"];
  auto & lanes = input_data_map["lanes"];
  auto & lanes_speed_limit = input_data_map["lanes_speed_limit"];
  auto & route_lanes = input_data_map["route_lanes"];

  // Allocate raw memory for bool array
  size_t lane_speed_tensor_num_elements = std::accumulate(
    LANES_SPEED_LIMIT_SHAPE.begin(), LANES_SPEED_LIMIT_SHAPE.end(), 1, std::multiplies<>());
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
    return session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 7, output_names, 1);
  } catch (const Ort::Exception & e) {
    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    return std::nullopt;
  }
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

  publish_debug_markers(input_data_map);

  // normalization of data
  preprocess::normalize_input_data(input_data_map, normalization_map_);
  if (!utils::check_input_map(input_data_map)) {
    RCLCPP_WARN(get_logger(), "Input data contains invalid values");
    return;
  }

  auto output = do_inference(input_data_map);
  if (!output) {
    return;
  }
  auto & predictions = output.value()[0];
  publish_predictions(predictions);
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
    preprocess::process_segments_to_matrix(lane_segments_, col_id_mapping_);

  route_handler_->setMap(*map_msg);
  is_map_loaded_ = true;
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
