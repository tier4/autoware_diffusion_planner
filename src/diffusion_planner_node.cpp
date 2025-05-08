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

#include <autoware_utils/math/normalization.hpp>

#include <Eigen/src/Core/Matrix.h>

#include <cassert>
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
  pub_route_marker_ = node->create_publisher<MarkerArray>("~/debug/route_marker", 1);
  pub_lane_marker_ = node->create_publisher<MarkerArray>("~/debug/lane_marker", 1);
  debug_processing_time_detail_pub_ = node->create_publisher<autoware_utils::ProcessingTimeDetail>(
    "~/debug/processing_time_detail_ms", 1);
  time_keeper_ = std::make_shared<autoware_utils::TimeKeeper>(debug_processing_time_detail_pub_);

  set_up_params();
  // Load the model
  if (params_.model_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Model path is not set");
    return;
  }

  normalization_map_ = load_normalization_stats(params_.args_path);
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
  params_.args_path = this->declare_parameter<std::string>("args_path", "");
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

AgentData DiffusionPlanner::get_ego_centric_agent_data(
  const TrackedObjects & objects, const Eigen::Matrix4f & map_to_ego_transform)
{
  if (!agent_data_) {
    agent_data_ =
      AgentData(objects, neighbor_agents_past_shape_[1], neighbor_agents_past_shape_[2]);
  } else {
    agent_data_->update_histories(objects);
  }

  auto ego_centric_data = agent_data_.value();
  ego_centric_data.apply_transform(map_to_ego_transform);
  ego_centric_data.trim_to_k_closest_agents();
  return ego_centric_data;
}

std::vector<float> DiffusionPlanner::extract_ego_centric_lane_segments(
  const Eigen::MatrixXf & ego_centric_lane_segments)
{
  const auto total_lane_points = lanes_shape_[1] * NUM_LANE_POINTS;
  Eigen::MatrixXf lane_segments_matrix(total_lane_points, LANE_POINT_DIM);
  lane_segments_matrix.block(0, 0, total_lane_points, LANE_POINT_DIM) =
    ego_centric_lane_segments.block(0, 0, total_lane_points, LANE_POINT_DIM);
  // convert to vector
  // print for debug
  // std::cerr << "ego_centric_lane_segments\n";
  // for (long i = 0; i < lane_segments_matrix.rows(); ++i) {
  //   for (long j = 0; j < lane_segments_matrix.cols(); ++j) {
  //     std::cerr << lane_segments_matrix(i, j) << " ";
  //   }
  //   std::cerr << std::endl;
  // }
  return {lane_segments_matrix.data(), lane_segments_matrix.data() + lane_segments_matrix.size()};
}

std::vector<float> DiffusionPlanner::extract_lane_speeds(
  const Eigen::MatrixXf & ego_centric_lane_segments)
{
  const auto total_lane_points = lanes_speed_limit_shape_[1] * NUM_LANE_POINTS;
  Eigen::MatrixXf lane_segments_speed(total_lane_points, lanes_speed_limit_shape_[2]);
  lane_segments_speed.block(0, 0, total_lane_points, lanes_speed_limit_shape_[2]) =
    ego_centric_lane_segments.block(0, 12, total_lane_points, lanes_speed_limit_shape_[2]);
  return {lane_segments_speed.data(), lane_segments_speed.data() + lane_segments_speed.size()};
}

std::vector<float> DiffusionPlanner::get_route_segments(
  const Eigen::Matrix4f & map_to_ego_transform)
{
  Eigen::MatrixXf full_route_segment_matrix(
    NUM_LANE_POINTS * route_ptr_->segments.size(), LANE_MATRIX_DIM);
  long route_segment_rows = 0;
  for (const auto & route_segment : route_ptr_->segments) {
    auto route_segment_row = segment_row_indices_[route_segment.preferred_primitive.id];
    full_route_segment_matrix.block(route_segment_rows, 0, NUM_LANE_POINTS, LANE_MATRIX_DIM) =
      map_lane_segments_matrix_.block(route_segment_row, 0, NUM_LANE_POINTS, LANE_MATRIX_DIM);
    route_segment_rows += NUM_LANE_POINTS;
  }

  Eigen::MatrixXf ego_centric_route_segments = transform_and_select_rows(
    full_route_segment_matrix, map_to_ego_transform, route_lanes_shape_[1]);

  // std::cerr << " ego_centric_route_segments\n";
  // for (long i = 0; i < ego_centric_route_segments.rows(); ++i) {
  //   for (long j = 0; j < ego_centric_route_segments.cols(); ++j) {
  //     std::cerr << ego_centric_route_segments(i, j) << " ";
  //   }
  //   std::cerr << std::endl;
  // }

  const auto total_route_points = route_lanes_shape_[1] * NUM_LANE_POINTS;
  Eigen::MatrixXf route_segments_matrix(total_route_points, LANE_POINT_DIM);
  route_segments_matrix.block(0, 0, total_route_points, LANE_POINT_DIM) =
    ego_centric_route_segments.block(0, 0, total_route_points, LANE_POINT_DIM);
  return {
    route_segments_matrix.data(), route_segments_matrix.data() + route_segments_matrix.size()};
}

void DiffusionPlanner::normalize_input_data(InputDataMap & input_data_map)
{
  auto normalize_vector = [](
                            std::vector<float> & data, const std::vector<float> & mean,
                            const std::vector<float> & std_dev) -> void {
    assert(!data.empty() && "Data vector must not be empty");
    assert((mean.size() == std_dev.size()) && "Mean and std must be same size");
    assert((data.size() % std_dev.size() == 0) && "Data size must be divisible by std_dev size");
    auto cols = std_dev.size();
    auto rows = data.size() / cols;

    for (size_t row = 0; row < rows; ++row) {
      const auto offset = row * cols;
      const auto row_begin =
        data.begin() + static_cast<std::vector<float>::difference_type>(offset);

      bool is_zero_row = std::all_of(
        row_begin, row_begin + static_cast<std::vector<float>::difference_type>(cols),
        [](float x) { return std::abs(x) < std::numeric_limits<float>::epsilon(); });

      if (is_zero_row) continue;

      for (size_t col = 0; col < cols; ++col) {
        float m = (mean.size() == 1) ? mean[0] : mean[col];
        float s = (std_dev.size() == 1) ? std_dev[0] : std_dev[col];
        data[offset + col] = (data[offset + col] - m) / s;
      }
    }
  };

  for (auto & [key, value] : input_data_map) {
    if (normalization_map_.find(key) != normalization_map_.end()) {
      const auto & [mean, std_dev] = normalization_map_[key];
      normalize_vector(value, mean, std_dev);
    } else {
      RCLCPP_WARN(get_logger(), "No normalization data for key: %s", key.c_str());
    }
  }
}

MarkerArray DiffusionPlanner::create_route_marker(
  const std::vector<float> & route_vector, [[maybe_unused]] const std::vector<long> & shape,
  const Time & stamp, const std::array<float, 4> colors, std::string ns)
{
  MarkerArray marker_array;
  // const long B = shape[0];  // batch size (should be 1)
  // const long L = shape[1];  // number of lanes = 25
  // const long P = 20;        // number of points = 20
  // const long D = 14;        // features per point = 12

  // if (B != 1) {
  //   throw std::runtime_error("Only batch size of 1 is supported.");
  // }

  // auto get_value = [&](long b, long l, long p, long d) -> float {
  //   long index = ((b * L + l) * P + p) * D + d;
  //   return route_vector[index];
  // };

  for (long l = 0; l < map_lane_segments_matrix_.rows() / 20; ++l) {
    // Check if the centerline is all zeros
    Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = ns;
    marker.ns = "route";
    marker.id = static_cast<int>(l);
    marker.type = Marker::LINE_STRIP;
    marker.action = Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.6;

    ColorRGBA color;
    color.r = colors[0];
    color.g = colors[1];
    color.b = colors[2];
    color.a = colors[3];
    marker.color = color;

    Duration lifetime;
    lifetime.sec = 1;
    lifetime.nanosec = 0;
    marker.lifetime = lifetime;

    for (long p = 0; p < 20; ++p) {
      auto x = route_vector[20 * l + p * 14 + 0];
      auto y = route_vector[20 * l + p * 14 + 1];
      float z = 0.5f;
      float norm = std::sqrt(x * x + y * y + z * z);
      if (norm < 2.0f) continue;

      Point pt;
      pt.x = x;
      pt.y = y;
      pt.z = z;

      marker.points.push_back(pt);
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
  route_ptr_ = (!route_ptr_ || temp_route) ? temp_route : route_ptr_;
  if (!is_map_loaded_) {
    RCLCPP_INFO(get_logger(), "Waiting for map data...");
    return {};
  }

  if (!objects || !ego_kinematic_state || !ego_acceleration || !route_ptr_) {
    RCLCPP_WARN(get_logger(), "No tracked objects or ego kinematic state or route data received");
    return {};
  }
  // Ego state
  // TODO(Daniel): use vehicle_info_utils
  EgoState ego_state(*ego_kinematic_state, *ego_acceleration, 5.0);
  input_data_map["ego_current_state"] = ego_state.as_array();

  // Agent data on ego reference frame
  transforms_ = get_transform_matrix(*ego_kinematic_state);
  auto map_to_ego_transform = transforms_.second;
  auto ego_centric_data = get_ego_centric_agent_data(*objects, map_to_ego_transform);
  input_data_map["neighbor_agents_past"] = ego_centric_data.as_vector();

  // Static objects
  // TODO(Daniel): add static objects
  auto static_objects = create_float_data(static_objects_shape_, 0.0f);
  input_data_map["static_objects"] = static_objects;

  // map data on ego reference frame
  Eigen::MatrixXf ego_centric_lane_segments =
    transform_and_select_rows(map_lane_segments_matrix_, map_to_ego_transform, lanes_shape_[1]);
  auto lane_data = extract_ego_centric_lane_segments(ego_centric_lane_segments);
  input_data_map["lanes"] = lane_data;
  auto lane_speed_data = extract_lane_speeds(ego_centric_lane_segments);
  input_data_map["lanes_speed_limit"] = lane_speed_data;

  // route data on ego reference frame
  auto route_segments = get_route_segments(map_to_ego_transform);
  input_data_map["route_lanes"] = route_segments;

  // auto route_markers = create_route_marker(route_segments, route_lanes_shape_, this->now());
  // pub_route_marker_->publish(route_markers);

  // normalization of data
  normalize_input_data(input_data_map);
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
  for (auto & dim : prediction_shape) {
    std::cerr << "Prediction shape: " << dim << " ";
  }
  std::cerr << std::endl;
  // const auto num_points_in_one_mode =
  //   prediction_shape[num_of_dimensions - 1] * prediction_shape[num_of_dimensions - 2];
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

  // print matrix for debugging
  // std::cerr << "Prediction matrix:\n";
  // for (long i = 0; i < prediction_matrix.rows(); ++i) {
  //   for (long j = 0; j < prediction_matrix.cols(); ++j) {
  //     std::cerr << prediction_matrix(i, j) << " ";
  //   }
  //   std::cerr << std::endl;
  // }

  return trajectory;
}

void DiffusionPlanner::on_timer()
{
  // Timer callback function
  autoware_utils::ScopedTimeTrack st(__func__, *time_keeper_);

  // Prepare input data for the model
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session_, mem_info);

  auto input_data_map = create_input_data();
  if (input_data_map.empty()) {
    RCLCPP_WARN(get_logger(), "No input data available for inference");
    return;
  }

  std::vector<float> lane_data(
    map_lane_segments_matrix_.data(),
    map_lane_segments_matrix_.data() + map_lane_segments_matrix_.size());
  const std::vector<long> map_lanes_shape_ = {1, map_lane_segments_matrix_.rows() / 20, 20, 14};

  for (long l = 0; l < map_lane_segments_matrix_.rows() / 20; ++l) {
    for (long p = 0; p < 20; ++p) {
      auto x = lane_data[20 * l + p * 14 + 0];
      auto y = lane_data[20 * l + p * 14 + 1];
      std::cerr << "x: " << x << " y: " << y << "\n";
      assert(x > 100.0 && "GOD NO X");
      assert(y > 100.0 && "GOD NO Y");
    }
  }

  auto lane_markers =
    create_route_marker(lane_data, map_lanes_shape_, this->now(), {0.0f, 0.0f, 1.0f, 0.8f}, "map");
  pub_lane_marker_->publish(lane_markers);

  auto ego_current_state = input_data_map["ego_current_state"];
  auto neighbor_agents_past = input_data_map["neighbor_agents_past"];
  auto static_objects = input_data_map["static_objects"];
  auto lanes = input_data_map["lanes"];
  auto lanes_speed_limit = input_data_map["lanes_speed_limit"];
  auto route_lanes = input_data_map["route_lanes"];

  // Allocate raw memory for bool array
  size_t lane_speed_tensor_num_elements = std::accumulate(
    lanes_speed_limit_shape_.begin(), lanes_speed_limit_shape_.end(), 1,
    std::multiplies<int64_t>());
  auto raw_speed_bool_array =
    std::shared_ptr<bool>(new bool[lane_speed_tensor_num_elements], std::default_delete<bool[]>());

  for (size_t i = 0; i < lane_speed_tensor_num_elements; ++i) {
    raw_speed_bool_array.get()[i] = (lanes_speed_limit[i] > std::numeric_limits<float>::epsilon());
  }

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
  auto lane_has_speed_limit_tensor = Ort::Value::CreateTensor<bool>(
    mem_info, raw_speed_bool_array.get(), lane_speed_tensor_num_elements,
    lanes_has_speed_limit_shape_.data(), lanes_has_speed_limit_shape_.size());
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

    auto output_trajectory = create_trajectory(output, transforms_.first);
    pub_trajectory_->publish(output_trajectory);
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

  map_lane_segments_matrix_ = lanelet_converter_ptr_->process_segments_to_matrix(
    lane_segments_, segment_row_indices_, 0.0, 0.0, 100000000.0);

  is_map_loaded_ = true;
  // std::cerr << "Lane segments matrix: " << map_lane_segments_matrix_.rows() << "x"
  //           << map_lane_segments_matrix_.cols() << std::endl;

  // for (long i = 0; i < map_lane_segments_matrix_.rows(); ++i) {
  //   for (long j = 0; j < map_lane_segments_matrix_.cols(); ++j) {
  //     std::cerr << map_lane_segments_matrix_(i, j) << " ";
  //   }
  //   std::cerr << std::endl;
  // }
}

void DiffusionPlanner::on_parameter(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
}

}  // namespace autoware::diffusion_planner
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::diffusion_planner::DiffusionPlanner)
