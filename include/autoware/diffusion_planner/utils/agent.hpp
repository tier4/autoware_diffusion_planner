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

#ifndef AUTOWARE__DIFFUSION_PLANNER__AGENT_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__AGENT_HPP_

#include "autoware/diffusion_planner/utils/fixed_queue.hpp"
#include "autoware/object_recognition_utils/object_recognition_utils.hpp"
#include "autoware_utils_geometry/geometry.hpp"

#include <autoware_perception_msgs/msg/tracked_object.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
namespace autoware::diffusion_planner
{
using autoware_perception_msgs::msg::TrackedObject;

constexpr size_t AgentStateDim = 11;

enum AgentLabel { VEHICLE = 0, PEDESTRIAN = 1, BICYCLE = 2 };

enum AgentDimLabels {
  X = 0,
  Y = 1,
  COS_YAW = 2,
  SIN_YAW = 3,
  VX = 4,
  VY = 5,
  L = 6,
  W = 7,
  LABEL_VEHICLE = 8,
  LABEL_PEDESTRIAN = 9,
  LABEL_BICYCLE = 10,
};

AgentLabel get_model_label(const autoware_perception_msgs::msg::TrackedObject & object)
{
  auto autoware_label =
    autoware::object_recognition_utils::getHighestProbLabel(object.classification);

  switch (autoware_label) {
    case autoware_perception_msgs::msg::ObjectClassification::CAR:
    case autoware_perception_msgs::msg::ObjectClassification::TRUCK:
    case autoware_perception_msgs::msg::ObjectClassification::BUS:
    case autoware_perception_msgs::msg::ObjectClassification::MOTORCYCLE:
    case autoware_perception_msgs::msg::ObjectClassification::TRAILER:
      return AgentLabel::VEHICLE;
    case autoware_perception_msgs::msg::ObjectClassification::BICYCLE:
      return AgentLabel::BICYCLE;
    case autoware_perception_msgs::msg::ObjectClassification::PEDESTRIAN:
      return AgentLabel::PEDESTRIAN;
    default:
      return AgentLabel::VEHICLE;
  }
  return AgentLabel::VEHICLE;
}

/**
 * @brief A class to represent a single state of an agent.
 */
struct AgentState
{
  // Construct a new instance filling all elements by `0.0f`.
  AgentState() = default;

  /**
   * @brief Construct a new instance with specified values.
   *
   * @param position 3D position [m].
   * @param dimension Box dimension [m].
   * @param yaw Heading yaw angle [rad].
   * @param velocity Velocity [m/s].
   * @param label Agent label
   */

  explicit AgentState(TrackedObject & object)
  {
    position_ = object.kinematics.pose_with_covariance.pose.position;
    dimension_ = object.shape.dimensions;
    float yaw =
      autoware_utils_geometry::get_rpy(object.kinematics.pose_with_covariance.pose.orientation).z;
    cos_yaw_ = std::cos(yaw);
    sin_yaw_ = std::sin(yaw);
    velocity_ = object.kinematics.twist_with_covariance.twist.linear;
    label_ = get_model_label(object);
  }

  AgentState(
    const geometry_msgs::msg::Point & position, const geometry_msgs::msg::Vector3 & dimension,
    float yaw, const geometry_msgs::msg::Vector3 & velocity, const AgentLabel & label)
  : position_(position),
    dimension_(dimension),
    cos_yaw_(std::cos(yaw)),
    sin_yaw_(std::sin(yaw)),
    velocity_(velocity),
    label_(label)
  {
  }

  // Construct a new instance filling all elements by `0.0f`.
  static AgentState empty() noexcept { return {}; }

  // Return the agent state dimensions `D`.
  static size_t dim() { return AgentStateDim; }

  // Return the x position.
  float x() const { return position_.x; }

  // Return the y position.
  float y() const { return position_.y; }

  // Return the z position.
  float z() const { return position_.z; }

  // Return the length of object size.
  float length() const { return dimension_.x; }

  // Return the width of object size.
  float width() const { return dimension_.y; }

  // Return the cos of yaw.
  float cos_yaw() const { return cos_yaw_; }

  // Return the sin of yaw.
  float sin_yaw() const { return sin_yaw_; }

  // Return the x velocity.
  float vx() const { return velocity_.x; }

  // Return the y velocity.
  float vy() const { return velocity_.y; }

  // Return the state attribute as an array.
  std::array<float, AgentStateDim> as_array() const noexcept
  {
    return {
      x(),
      y(),
      cos_yaw(),
      sin_yaw(),
      vx(),
      vy(),
      length(),
      width(),
      static_cast<float>(label_ == AgentLabel::VEHICLE),
      static_cast<float>(label_ == AgentLabel::PEDESTRIAN),
      static_cast<float>(label_ == AgentLabel::BICYCLE),
    };
  }

  geometry_msgs::msg::Point position_;
  geometry_msgs::msg::Vector3 dimension_;
  float cos_yaw_{0.0f};
  float sin_yaw_{0.0f};
  geometry_msgs::msg::Vector3 velocity_;
  AgentLabel label_{AgentLabel::VEHICLE};
};

/**
 * @brief A class to represent the state history of an agent.
 */
struct AgentHistory
{
  /**
   * @brief Construct a new Agent History filling the latest state by input state.
   *
   * @param state Object current state.
   * @param object_id Object ID.
   * @param label_id Label ID.
   * @param current_time Current timestamp.
   * @param max_time_length History length.
   */
  AgentHistory(
    const AgentState & state, std::string object_id, const size_t label_id,
    const double current_time, const size_t max_time_length)
  : queue_(max_time_length),
    object_id_(std::move(object_id)),
    label_id_(label_id),
    latest_time_(current_time),
    max_time_length_(max_time_length)
  {
    queue_.push_back(state);
  }

  // Return the history time length `T`.
  [[nodiscard]] size_t length() const { return max_time_length_; }

  // Return the number of agent state dimensions `D`.
  static size_t state_dim() { return AgentStateDim; }

  // Return the data size of history `T * D`.
  [[nodiscard]] size_t size() const { return max_time_length_ * state_dim(); }

  // Return the shape of history matrix ordering in `(T, D)`.
  [[nodiscard]] std::tuple<size_t, size_t> shape() const { return {max_time_length_, state_dim()}; }

  // Return the object id.
  [[nodiscard]] const std::string & object_id() const { return object_id_; }

  // Return the label id.
  [[nodiscard]] size_t label_id() const { return label_id_; }

  /**
   * @brief Return the last timestamp when non-empty state was pushed.
   *
   * @return double
   */
  [[nodiscard]] double latest_time() const { return latest_time_; }

  /**
   * @brief Update history with input state and latest time.
   *
   * @param current_time The current timestamp.
   * @param object The object info.
   */
  void update(double current_time, TrackedObject & object) noexcept
  {
    AgentState state(object);
    queue_.push_back(state);
    latest_time_ = current_time;
  }

  /**
   * @brief Update history with input state and latest time.
   *
   * @param current_time The current timestamp.
   * @param state The current agent state.
   */
  void update(double current_time, const AgentState & state) noexcept
  {
    queue_.push_back(state);
    latest_time_ = current_time;
  }

  // Update history with all-zeros state, but latest time is not updated.
  void update_empty() noexcept
  {
    const auto state = AgentState::empty();
    queue_.push_back(state);
  }

  // Return a history states as an array.
  [[nodiscard]] std::vector<float> as_array() const noexcept
  {
    std::vector<float> output;
    for (const auto & state : queue_) {
      for (const auto & v : state.as_array()) {
        output.push_back(v);
      }
    }
    return output;
  }

  /**
   * @brief Check whether the latest valid state is too old or not.
   *
   * @param current_time Current timestamp.
   * @param threshold Time difference threshold value.
   * @return true If the difference is greater than threshold.
   * @return false Otherwise
   */
  [[nodiscard]] bool is_ancient(double current_time, double threshold) const
  {
    /* TODO: Raise error if the current time is smaller than latest */
    return current_time - latest_time_ >= threshold;
  }

  // Get the latest agent state at `T`.
  [[nodiscard]] const AgentState & get_latest_state() const { return queue_.back(); }

  [[nodiscard]] const geometry_msgs::msg::Point & get_latest_state_position() const
  {
    return get_latest_state().position_;
  }

  // private:
  FixedQueue<AgentState> queue_;
  std::string object_id_;
  size_t label_id_;
  double latest_time_;
  size_t max_time_length_;
};

/**
 * @brief A class containing whole state histories of all agent.
 */
struct AgentData
{
  /**
   * @brief Construct a new instance.
   *
   * @param histories An array of histories for each object.
   * @param num_agent Number of agents.
   * @param num_timestamps Number of timestamps.
   */
  AgentData(
    const std::vector<AgentHistory> & histories, const size_t num_agent,
    const size_t num_timestamps)
  : histories_(histories), num_agent_(num_agent), time_length_(num_timestamps)
  {
    fill_data(histories);
  }

  // fill data array
  void fill_data(const std::vector<AgentHistory> & histories)
  {
    data_.clear();
    data_.reserve(num_agent_ * time_length_ * state_dim());
    for (auto & history : histories) {
      for (const auto & v : history.as_array()) {
        data_.push_back(v);
      }
    }
  }
  // Return the number of classes `C`.
  static size_t num_class() { return 3; }

  // Return the number of agents `N`.
  size_t num_agent() const { return num_agent_; }

  // Return the timestamp length `T`.
  size_t time_length() const { return time_length_; }

  // Return the number of agent state dimensions `D`.
  static size_t state_dim() { return AgentStateDim; }

  // Return the number of all elements `N*T*D`.
  size_t size() const { return num_agent_ * time_length_ * state_dim(); }

  // Return the number of state dimensions of MTR input `T+C+D+3`.
  size_t input_dim() const { return time_length_ + state_dim() + num_class() + 3; }

  // Return the data shape ordering in (N, T, D).
  std::tuple<size_t, size_t, size_t> shape() const
  {
    return {num_agent_, time_length_, state_dim()};
  }

  // Return the address pointer of data array.
  const float * data_ptr() const noexcept { return data_.data(); }

private:
  std::vector<AgentHistory> histories_;
  size_t num_agent_;
  size_t time_length_;
  std::vector<float> data_;
};

}  // namespace autoware::diffusion_planner
#endif  // AUTOWARE__DIFFUSION_PLANNER__AGENT_HPP_
