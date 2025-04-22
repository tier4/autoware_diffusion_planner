// Copyright 2024 TIER IV, Inc.
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

#include "onnxruntime_cxx_api.h"

#include <cstdint>
#include <iostream>
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
}

void DiffusionPlanner::set_up_params()
{
  params_.model_path = this->declare_parameter<std::string>("onnx_model_path", "");
  params_.planning_frequency_hz = this->declare_parameter<double>("planning_frequency_hz", 10.0);
  RCLCPP_INFO(get_logger(), "Setting up parameters for Diffusion Planner");
}

void DiffusionPlanner::load_model(const std::string & model_path)
{
  // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DiffusionPlanner");
  // Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "DiffusionPlanner");
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
  RCLCPP_INFO(get_logger(), "Diffusion Planner Timer Callback");
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session_, mem_info);

  const std::vector<int64_t> ego_current_state_shape = {1, 10};
  const std::vector<int64_t> neighbor_agents_past_shape = {1, 32, 21, 11};
  const std::vector<int64_t> lane_has_speed_limit_shape = {1, 70, 1};
  const std::vector<int64_t> static_objects_shape = {1, 5, 10};
  const std::vector<int64_t> lanes_shape = {1, 70, 20, 12};
  const std::vector<int64_t> lanes_speed_limit_shape = {1, 70, 1};
  const std::vector<int64_t> route_lanes_shape = {1, 25, 20, 12};

  auto ego_current_state = create_float_data(ego_current_state_shape);
  auto neighbor_agents_past = create_float_data(neighbor_agents_past_shape);
  auto static_objects = create_float_data(static_objects_shape);
  auto lanes = create_float_data(lanes_shape);
  auto lanes_speed_limit = create_float_data(lanes_speed_limit_shape);
  auto route_lanes = create_float_data(route_lanes_shape);

  auto ego_current_state_tensor = Ort::Value::CreateTensor<float>(
    mem_info, ego_current_state.data(), ego_current_state.size(), ego_current_state_shape.data(),
    ego_current_state_shape.size());
  auto neighbor_agents_past_tensor = Ort::Value::CreateTensor<float>(
    mem_info, neighbor_agents_past.data(), neighbor_agents_past.size(),
    neighbor_agents_past_shape.data(), neighbor_agents_past_shape.size());
  auto static_objects_tensor = Ort::Value::CreateTensor<float>(
    mem_info, static_objects.data(), static_objects.size(), static_objects_shape.data(),
    static_objects_shape.size());
  auto lanes_tensor = Ort::Value::CreateTensor<float>(
    mem_info, lanes.data(), lanes.size(), lanes_shape.data(), lanes_shape.size());
  auto lanes_speed_limit_tensor = Ort::Value::CreateTensor<float>(
    mem_info, lanes_speed_limit.data(), lanes_speed_limit.size(), lanes_speed_limit_shape.data(),
    lanes_speed_limit_shape.size());

  size_t total_size = 70;
  std::vector<int64_t> shape = {1, 70, 1};
  std::vector<std::shared_ptr<void>> keep_alive_blobs_;
  // Allocate raw memory for bool array
  auto raw_bool_array = std::shared_ptr<bool>(new bool[total_size], std::default_delete<bool[]>());
  // Initialize with true values
  std::fill(raw_bool_array.get(), raw_bool_array.get() + total_size, true);
  // Create the tensor
  auto lane_has_speed_limit_tensor = Ort::Value::CreateTensor<bool>(
    mem_info, raw_bool_array.get(), total_size, shape.data(), shape.size());
  // Ensure the tensor's data is kept alive
  keep_alive_blobs_.emplace_back(raw_bool_array);
  auto route_lanes_tensor = Ort::Value::CreateTensor<float>(
    mem_info, route_lanes.data(), route_lanes.size(), route_lanes_shape.data(),
    route_lanes_shape.size());

  Ort::Value input_tensors[] = {
    std::move(ego_current_state_tensor), std::move(neighbor_agents_past_tensor),
    std::move(static_objects_tensor),    std::move(lanes_tensor),
    std::move(lanes_speed_limit_tensor), std::move(lane_has_speed_limit_tensor),
    std::move(route_lanes_tensor)};

  const char * input_names[] = {
    "ego_current_state", "neighbor_agents_past",  "static_objects", "lanes",
    "lanes_speed_limit", "lanes_has_speed_limit", "route_lanes"};

  const char * output_names[] = {"output"};

  for (size_t i = 0; i < 7; ++i) {
    std::cout << "Input " << i << " name: " << input_names[i] << std::endl;
    std::cout << "IsTensor: " << input_tensors[i].IsTensor() << std::endl;
    std::cout << "Tensor type: " << input_tensors[i].GetTensorTypeAndShapeInfo().GetElementType()
              << std::endl;
    std::cout << "Tensor shape: ";
    for (auto dim : input_tensors[i].GetTensorTypeAndShapeInfo().GetShape()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
  // run inference
  std::cerr << "Running inference..." << std::endl;
  try {
    auto output =
      session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 7, output_names, 1);
    std::cout << "Inference ran successfully, got " << output.size() << " outputs." << std::endl;
  } catch (const Ort::Exception & e) {
    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
  }

  for (size_t i = 0; i < 7; ++i) {
    std::cout << "Input " << i << " name: " << input_names[i] << std::endl;
    std::cout << "IsTensor: " << input_tensors[i].IsTensor() << std::endl;
    std::cout << "Tensor type: " << input_tensors[i].GetTensorTypeAndShapeInfo().GetElementType()
              << std::endl;
    std::cout << "Tensor shape: ";
    for (auto dim : input_tensors[i].GetTensorTypeAndShapeInfo().GetShape()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }
}

void DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
