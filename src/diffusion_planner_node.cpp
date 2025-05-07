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
#include "onnxruntime_cxx_api.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
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
  debug_processing_time_detail_pub_ = node->create_publisher<autoware_utils::ProcessingTimeDetail>(
    "~/debug/processing_time_detail_ms", 1);
  time_keeper_ = std::make_shared<autoware_utils::TimeKeeper>(debug_processing_time_detail_pub_);

  set_up_params();
  // Load the model
  if (params_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Model path is not set");
    return;
  }

  load_model(params_.model_path);

  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Rate(params_.planning_frequency_hz).period(),
    std::bind(&DiffusionPlanner::on_timer, this));

  sub_map_ = create_subscription<HADMapBin>(
    "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&DiffusionPlanner::on_map, this, std::placeholders::_1));
}

void DiffusionPlanner::set_up_params()
{
  params_.model_path = this->declare_parameter<std::string>("onnx_model_path", "");
  params_.planning_frequency_hz = this->declare_parameter<double>("planning_frequency_hz", 10.0);
  RCLCPP_INFO(get_logger(), "Setting up parameters for Diffusion Planner");
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

void DiffusionPlanner::on_timer()
{
  // Timer callback function
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);

  auto objects = sub_tracked_objects_.take_data();
  auto ego_kinematic_state = sub_current_odometry_.take_data();
  auto ego_acceleration = sub_current_acceleration_.take_data();

  if (!is_map_loaded_) {
    RCLCPP_INFO(get_logger(), "Waiting for map data...");
    return;
  }

  if (!objects || !ego_kinematic_state || !ego_acceleration) {
    RCLCPP_WARN(get_logger(), "No tracked objects or ego kinematic state data received");
    return;
  }

  EgoState ego_state(
    *ego_kinematic_state, *ego_acceleration, 5.0);  // TODO(Daniel): use vehicle_info_utils

  std::cerr << ego_state.to_string() << "\n";

  if (!agent_data_) {
    agent_data_ =
      AgentData(*objects, neighbor_agents_past_shape_[1], neighbor_agents_past_shape_[2]);
  } else {
    agent_data_->update_histories(*objects);
  }

  std::pair<Eigen::Matrix4f, Eigen::Matrix4f> transforms =
    get_transform_matrix(*ego_kinematic_state);
  auto ego_centric_data = agent_data_.value();
  auto map_to_ego_transform = transforms.second;
  ego_centric_data.apply_transform(map_to_ego_transform);
  geometry_msgs::msg::Point position;
  position.x = 0.0;
  position.y = 0.0;
  position.z = 0.0;
  ego_centric_data.trim_to_k_closest_agents(position);

  std::cerr << "Agent Data: " << ego_centric_data.to_string() << std::endl;

  Eigen::MatrixXf ego_centric_lane_segments =
    transform_and_select_rows(map_lane_segments_matrix_, map_to_ego_transform, lanes_shape_[1]);

  std::cerr << "FULL Ego centric Lane segments matrix: " << ego_centric_lane_segments.rows() << "x"
            << ego_centric_lane_segments.cols() << std::endl;

  for (long i = 0; i < ego_centric_lane_segments.rows(); ++i) {
    for (long j = 0; j < ego_centric_lane_segments.cols(); ++j) {
      std::cerr << ego_centric_lane_segments(i, j) << " ";
    }
    std::cerr << std::endl;
  }

  const auto total_lane_points = lanes_speed_limit_shape_[1] * NUM_LANE_POINTS;

  Eigen::MatrixXf lane_segments_matrix(total_lane_points, LANE_POINT_DIM);
  lane_segments_matrix.block(0, 0, total_lane_points, LANE_POINT_DIM) =
    ego_centric_lane_segments.block(0, 0, total_lane_points, LANE_POINT_DIM);

  std::cerr << "Ego centric Lane segments matrix: " << lane_segments_matrix.rows() << "x"
            << lane_segments_matrix.cols() << std::endl;

  for (long i = 0; i < lane_segments_matrix.rows(); ++i) {
    for (long j = 0; j < lane_segments_matrix.cols(); ++j) {
      std::cerr << lane_segments_matrix(i, j) << " ";
    }
    std::cerr << std::endl;
  }

  Eigen::MatrixXf lane_segments_speed(total_lane_points, lanes_speed_limit_shape_[2]);
  lane_segments_speed.block(0, 0, total_lane_points, lanes_speed_limit_shape_[2]) =
    ego_centric_lane_segments.block(0, 12, total_lane_points, lanes_speed_limit_shape_[2]);

  std::cerr << "Ego centric Lane velocity matrix: " << lane_segments_speed.rows() << "x"
            << lane_segments_speed.cols() << std::endl;

  for (long i = 0; i < lane_segments_speed.rows(); ++i) {
    for (long j = 0; j < lane_segments_speed.cols(); ++j) {
      std::cerr << lane_segments_speed(i, j) << " ";
    }
    std::cerr << std::endl;
  }

  // Prepare input data for the model
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session_, mem_info);

  auto ego_current_state = create_float_data(ego_current_state_shape_);
  auto neighbor_agents_past = create_float_data(neighbor_agents_past_shape_);
  auto static_objects = create_float_data(static_objects_shape_);
  auto lanes = create_float_data(lanes_shape_);
  auto lanes_speed_limit = create_float_data(lanes_speed_limit_shape_);
  auto route_lanes = create_float_data(route_lanes_shape_);

  auto ego_current_state_tensor = Ort::Value::CreateTensor<float>(
    mem_info, ego_current_state.data(), ego_current_state.size(), ego_current_state_shape_.data(),
    ego_current_state_shape_.size());
  auto neighbor_agents_past_tensor = Ort::Value::CreateTensor<float>(
    mem_info, neighbor_agents_past.data(), neighbor_agents_past.size(),
    neighbor_agents_past_shape_.data(), neighbor_agents_past_shape_.size());
  auto static_objects_tensor = Ort::Value::CreateTensor<float>(
    mem_info, static_objects.data(), static_objects.size(), static_objects_shape_.data(),
    static_objects_shape_.size());
  auto lanes_tensor = Ort::Value::CreateTensor<float>(
    mem_info, lanes.data(), lanes.size(), lanes_shape_.data(), lanes_shape_.size());
  auto lanes_speed_limit_tensor = Ort::Value::CreateTensor<float>(
    mem_info, lanes_speed_limit.data(), lanes_speed_limit.size(), lanes_speed_limit_shape_.data(),
    lanes_speed_limit_shape_.size());

  size_t total_size = std::accumulate(
    lane_has_speed_limit_shape_.begin(), lane_has_speed_limit_shape_.end(), 1,
    std::multiplies<int64_t>());
  std::vector<std::shared_ptr<void>> keep_alive_blobs_;
  // Allocate raw memory for bool array
  auto raw_bool_array = std::shared_ptr<bool>(new bool[total_size], std::default_delete<bool[]>());
  // Initialize with true values
  std::fill(raw_bool_array.get(), raw_bool_array.get() + total_size, true);
  // Create the tensor
  auto lane_has_speed_limit_tensor = Ort::Value::CreateTensor<bool>(
    mem_info, raw_bool_array.get(), total_size, lanes_has_speed_limit_shape_.data(),
    lanes_has_speed_limit_shape_.size());
  // Ensure the tensor's data is kept alive
  keep_alive_blobs_.emplace_back(raw_bool_array);
  auto route_lanes_tensor = Ort::Value::CreateTensor<float>(
    mem_info, route_lanes.data(), route_lanes.size(), route_lanes_shape_.data(),
    route_lanes_shape_.size());

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
  std::cerr << "Running inference..." << std::endl;
  try {
    auto output =
      session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 7, output_names, 1);
    std::cout << "Inference ran successfully, got " << output.size() << " outputs." << std::endl;
    // auto data = output[0].GetTensorMutableData<float>();
    std::cout << "Output data: ";
    // for (size_t i = 0; i < output[0].GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
    //   std::cout << data[i] << " ";
    // }
  } catch (const Ort::Exception & e) {
    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
  }
}

void DiffusionPlanner::on_map(const HADMapBin::ConstSharedPtr map_msg)
{
  std::cerr << "Received map message" << std::endl;
  lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(
    *map_msg, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);

  lanelet_converter_ptr_ = std::make_unique<LaneletConverter>(lanelet_map_ptr_, 100, 20, 100.0);
  lane_segments_ = lanelet_converter_ptr_->convert_to_lane_segments();

  if (lane_segments_.empty()) {
    RCLCPP_WARN(get_logger(), "No lane segments found in the map");
    throw std::runtime_error("No lane segments found in the map");
  }

  map_lane_segments_matrix_ =
    lanelet_converter_ptr_->process_segments_to_matrix(lane_segments_, 0.0, 0.0, 100000000.0);

  is_map_loaded_ = true;
  std::cerr << "Lane segments matrix: " << map_lane_segments_matrix_.rows() << "x"
            << map_lane_segments_matrix_.cols() << std::endl;

  for (long i = 0; i < map_lane_segments_matrix_.rows(); ++i) {
    for (long j = 0; j < map_lane_segments_matrix_.cols(); ++j) {
      std::cerr << map_lane_segments_matrix_(i, j) << " ";
    }
    std::cerr << std::endl;
  }
}

void DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
