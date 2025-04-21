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
  timer_ = rclcpp::create_timer(
    this, get_clock(), rclcpp::Rate(params_.planning_frequency_hz).period(),
    std::bind(&DiffusionPlanner::on_timer, this));

  // Load the model
  if (params_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Model path is not set");
    return;
  }
  load_model(params_.model_path);

  // Load the model using ONNX Runtime
}

void DiffusionPlanner::set_up_params()
{
  params_.model_path = this->declare_parameter<std::string>("onnx_model_path", "");
  params_.planning_frequency_hz = this->declare_parameter<double>("planning_frequency_hz", 10.0);
  RCLCPP_INFO(get_logger(), "Setting up parameters for Diffusion Planner");
}

void DiffusionPlanner::load_model(const std::string & model_path)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DiffusionPlanner");
  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions cuda_options;
  session_options.SetLogSeverityLevel(1);
  session_options.AppendExecutionProvider_CUDA(cuda_options);
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
  session_ = Ort::Session(env, model_path.c_str(), session_options);
  RCLCPP_INFO(get_logger(), "Model loaded from %s", params_.model_path.c_str());
}

void DiffusionPlanner::on_timer()
{
  // Timer callback function
  RCLCPP_INFO(get_logger(), "Diffusion Planner Timer Callback");
  Ort::AllocatorWithDefaultOptions allocator;

  size_t num_inputs = session_.GetInputCount();
  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_name = session_.GetInputNameAllocated(i, allocator);
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> shape = tensor_info.GetShape();

    std::cout << "Input " << i << " : name=" << input_name.get() << ", type=" << type
              << ", shape=[";
    for (size_t j = 0; j < shape.size(); ++j)
      std::cout << shape[j] << (j + 1 < shape.size() ? ", " : "");
    std::cout << "]" << std::endl;
  }

  std::map<std::string, Ort::Value> inputs;

  size_t total_size = 70;
  std::shared_ptr<bool[]> plain_data(new bool[total_size]);
  std::vector<int64_t> shape = {1, 70, 1};

  // Use vector of bool to initialize
  std::vector<bool> bool_data(total_size, true);

  // Convert to plain bool array (since vector<bool> is special and not usable directly)
  auto it = bool_data.begin();
  for (size_t i = 0; i < total_size; ++i) {
    plain_data[i] = *(it++);
  }

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  // Create the inputs map
  std::map<std::string, Ort::Value> inputs_map;
  auto ego_current_state = create_float_data({1, 10});
  auto neighbor_agents_past = create_float_data({1, 32, 21, 11});
  auto static_objects = create_float_data({1, 5, 10});
  auto lanes = create_float_data({1, 70, 20, 12});
  auto lanes_speed_limit = create_float_data({1, 70, 1});
  auto route_lanes = create_float_data({1, 25, 20, 12});

  inputs_map.emplace("ego_current_state", create_tensor_float(ego_current_state, {1, 10}));
  inputs_map.emplace(
    "neighbor_agents_past", create_tensor_float(neighbor_agents_past, {1, 32, 21, 11}));
  inputs_map.emplace("static_objects", create_tensor_float(static_objects, {1, 5, 10}));
  inputs_map.emplace("lanes", create_tensor_float(lanes, {1, 70, 20, 12}));
  inputs_map.emplace("lanes_speed_limit", create_tensor_float(lanes_speed_limit, {1, 70, 1}));
  inputs_map.emplace("route_lanes", create_tensor_float(route_lanes, {1, 25, 20, 12}));

  // Make sure to construct lane_has_speed_limit separately
  // Define this at a scope that survives beyond the session.Run()
  // Allocate as uint8_t
  // Allocate a raw bool array on the heap
  std::vector<std::shared_ptr<void>> keep_alive_blobs_;

  // Allocate raw memory for bool array
  auto raw_bool_array = std::shared_ptr<bool>(new bool[total_size], std::default_delete<bool[]>());

  // Initialize with true values
  std::fill(raw_bool_array.get(), raw_bool_array.get() + total_size, true);

  // Create the tensor
  Ort::Value lane_has_speed_limit_tensor = Ort::Value::CreateTensor<bool>(
    mem_info, raw_bool_array.get(), total_size, shape.data(), shape.size());

  // Ensure the tensor's data is kept alive
  keep_alive_blobs_.emplace_back(raw_bool_array);

  // Add to input map
  inputs_map.emplace("lanes_has_speed_limit", std::move(lane_has_speed_limit_tensor));

  // Define the fixed input order
  std::vector<std::string> fixed_input_order = {
    "ego_current_state", "neighbor_agents_past",  "static_objects", "lanes",
    "lanes_speed_limit", "lanes_has_speed_limit", "route_lanes"};

  // Now build the name and tensor arrays safely
  std::vector<std::string> input_name_storage;
  std::vector<const char *> input_names;
  std::vector<Ort::Value> input_tensors;

  for (const auto & name : fixed_input_order) {
    input_name_storage.push_back(name);
    input_names.push_back(input_name_storage.back().c_str());
    auto node = inputs_map.extract(name);
    input_tensors.emplace_back(std::move(node.mapped()));
  }
  // prepare output names
  std::vector<const char *> output_names;
  std::vector<Ort::AllocatedStringPtr> output_name_holders;  // keeps strings alive
  size_t num_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name_ptr = session_.GetOutputNameAllocated(i, allocator);
    output_names.push_back(name_ptr.get());
    output_name_holders.push_back(std::move(name_ptr));
  }

  // run inference
  auto output = session_.Run(
    Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(),
    output_names.data(), output_names.size());

  std::cout << "Inference ran successfully, got " << output.size() << " outputs." << std::endl;
}

void DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
