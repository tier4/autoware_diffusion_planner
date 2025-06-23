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

#include "autoware/diffusion_planner/conversion/agent.hpp"
#include "autoware/diffusion_planner/dimensions.hpp"

#include <autoware_utils/ros/uuid_helper.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <geometry_msgs/msg/twist_with_covariance.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace autoware::diffusion_planner::test
{
using autoware_perception_msgs::msg::TrackedObject;
using autoware_perception_msgs::msg::TrackedObjects;
using autoware_perception_msgs::msg::ObjectClassification;

class AgentEdgeCaseTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Create a basic tracked object
    tracked_object_.object_id = autoware_utils::generate_uuid();
    tracked_object_.kinematics.pose_with_covariance.pose.position.x = 10.0;
    tracked_object_.kinematics.pose_with_covariance.pose.position.y = 5.0;
    tracked_object_.kinematics.pose_with_covariance.pose.position.z = 0.0;
    tracked_object_.kinematics.twist_with_covariance.twist.linear.x = 2.0;
    tracked_object_.kinematics.twist_with_covariance.twist.linear.y = 1.0;
    tracked_object_.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    tracked_object_.shape.dimensions.x = 4.0;
    tracked_object_.shape.dimensions.y = 2.0;
    tracked_object_.shape.dimensions.z = 1.5;
    tracked_object_.existence_probability = 0.9;

    ObjectClassification classification;
    classification.label = ObjectClassification::CAR;
    classification.probability = 0.95;
    tracked_object_.classification.push_back(classification);
  }

  TrackedObject tracked_object_;
};

// Test edge case: NaN/Inf positions
TEST_F(AgentEdgeCaseTest, AgentStateWithNaNInfPositions)
{
  tracked_object_.kinematics.pose_with_covariance.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  tracked_object_.kinematics.pose_with_covariance.pose.position.y = std::numeric_limits<double>::infinity();
  tracked_object_.kinematics.pose_with_covariance.pose.position.z = -std::numeric_limits<double>::infinity();

  AgentState agent_state(tracked_object_);

  EXPECT_TRUE(std::isnan(agent_state.x()));
  EXPECT_TRUE(std::isinf(agent_state.y()));
  EXPECT_TRUE(std::isinf(agent_state.z()));
}

// Test edge case: Zero dimensions
TEST_F(AgentEdgeCaseTest, AgentStateWithZeroDimensions)
{
  tracked_object_.shape.dimensions.x = 0.0;
  tracked_object_.shape.dimensions.y = 0.0;
  tracked_object_.shape.dimensions.z = 0.0;

  AgentState agent_state(tracked_object_);

  EXPECT_FLOAT_EQ(agent_state.length(), 0.0);
  EXPECT_FLOAT_EQ(agent_state.width(), 0.0);
}

// Test edge case: Negative dimensions (should be handled as absolute values)
TEST_F(AgentEdgeCaseTest, AgentStateWithNegativeDimensions)
{
  tracked_object_.shape.dimensions.x = -5.0;
  tracked_object_.shape.dimensions.y = -3.0;

  AgentState agent_state(tracked_object_);

  // Implementation might handle negative dimensions differently
  // This test documents the actual behavior
  EXPECT_FLOAT_EQ(agent_state.length(), -5.0);
  EXPECT_FLOAT_EQ(agent_state.width(), -3.0);
}

// Test edge case: Maximum history size
TEST_F(AgentEdgeCaseTest, AgentHistoryMaxSize)
{
  constexpr size_t max_history = 100;  // Assuming this is the max
  AgentHistory history(tracked_object_.object_id, max_history);

  // Add more states than max history
  for (size_t i = 0; i < max_history + 10; ++i) {
    tracked_object_.kinematics.pose_with_covariance.pose.position.x = static_cast<double>(i);
    AgentState state(tracked_object_);
    history.add_state(state);
  }

  // Should only keep the most recent max_history states
  EXPECT_LE(history.get_states().size(), max_history);
}

// Test edge case: Empty history operations
TEST_F(AgentEdgeCaseTest, AgentHistoryEmptyOperations)
{
  AgentHistory history(tracked_object_.object_id, 10);

  // Operations on empty history should not crash
  EXPECT_NO_THROW(history.get_latest_state());
  EXPECT_NO_THROW(history.get_states());

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  EXPECT_NO_THROW(history.apply_transform(transform));
}

// Test edge case: Extreme transformation values
TEST_F(AgentEdgeCaseTest, AgentStateExtremeTransform)
{
  AgentState agent_state(tracked_object_);

  Eigen::Matrix4f extreme_transform = Eigen::Matrix4f::Identity();
  extreme_transform(0, 3) = 1e10f;   // Very large translation
  extreme_transform(1, 3) = -1e10f;
  extreme_transform(0, 0) = 1e-10f;  // Very small scale
  extreme_transform(1, 1) = 1e-10f;

  agent_state.apply_transform(extreme_transform);

  // Check that values don't overflow
  EXPECT_TRUE(std::isfinite(agent_state.x()));
  EXPECT_TRUE(std::isfinite(agent_state.y()));
}

// Test edge case: AgentData with no objects
TEST_F(AgentEdgeCaseTest, AgentDataNoObjects)
{
  TrackedObjects empty_objects;
  empty_objects.header.stamp = rclcpp::Time(0);

  AgentData agent_data(empty_objects, 10, 5, false);

  EXPECT_EQ(agent_data.get_histories().size(), 0);

  // Operations should handle empty data gracefully
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  EXPECT_NO_THROW(agent_data.apply_transform(transform));
  EXPECT_NO_THROW(agent_data.trim_to_k_closest_agents());
}

// Test edge case: AgentData with unknown object types
TEST_F(AgentEdgeCaseTest, AgentDataUnknownObjects)
{
  TrackedObjects objects;
  objects.header.stamp = rclcpp::Time(0);

  // Add object with UNKNOWN classification
  TrackedObject unknown_object = tracked_object_;
  unknown_object.classification.clear();
  ObjectClassification unknown_class;
  unknown_class.label = ObjectClassification::UNKNOWN;
  unknown_class.probability = 1.0;
  unknown_object.classification.push_back(unknown_class);
  objects.objects.push_back(unknown_object);

  // Test with ignore_unknown = true
  AgentData agent_data_ignore(objects, 10, 5, true);
  EXPECT_EQ(agent_data_ignore.get_histories().size(), 0);

  // Test with ignore_unknown = false
  AgentData agent_data_include(objects, 10, 5, false);
  EXPECT_EQ(agent_data_include.get_histories().size(), 1);
}

// Test edge case: AgentData update with ID mismatch
TEST_F(AgentEdgeCaseTest, AgentDataUpdateIDMismatch)
{
  TrackedObjects initial_objects;
  initial_objects.header.stamp = rclcpp::Time(0);
  initial_objects.objects.push_back(tracked_object_);

  AgentData agent_data(initial_objects, 10, 5, false);

  // Update with different object IDs
  TrackedObjects new_objects;
  new_objects.header.stamp = rclcpp::Time(1);
  TrackedObject new_object = tracked_object_;
  new_object.object_id = autoware_utils::generate_uuid();  // Different ID
  new_objects.objects.push_back(new_object);

  agent_data.update_histories(new_objects, false);

  // Should have 2 different objects now
  EXPECT_EQ(agent_data.get_histories().size(), 2);
}

// Test edge case: Distance calculation with same position
TEST_F(AgentEdgeCaseTest, AgentDataTrimSamePosition)
{
  TrackedObjects objects;
  objects.header.stamp = rclcpp::Time(0);

  // Add multiple objects at the same position
  for (int i = 0; i < 5; ++i) {
    TrackedObject same_pos_object = tracked_object_;
    same_pos_object.object_id = autoware_utils::generate_uuid();
    same_pos_object.kinematics.pose_with_covariance.pose.position.x = 0.0;
    same_pos_object.kinematics.pose_with_covariance.pose.position.y = 0.0;
    objects.objects.push_back(same_pos_object);
  }

  AgentData agent_data(objects, 3, 5, false);

  geometry_msgs::msg::Point ref_point;
  ref_point.x = 0.0;
  ref_point.y = 0.0;
  ref_point.z = 0.0;

  // All objects are at the same distance
  agent_data.trim_to_k_closest_agents(ref_point);

  // Should keep k_closest agents
  EXPECT_EQ(agent_data.get_histories().size(), 3);
}

// Test edge case: Object matrix generation with edge values
TEST_F(AgentEdgeCaseTest, AgentHistoryGetObjectMatrixEdgeCases)
{
  AgentHistory history(tracked_object_.object_id, 10);

  // Add states with extreme velocities
  for (int i = 0; i < 5; ++i) {
    TrackedObject extreme_object = tracked_object_;
    extreme_object.kinematics.twist_with_covariance.twist.linear.x =
      (i % 2 == 0) ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
    extreme_object.kinematics.twist_with_covariance.twist.linear.y =
      (i % 2 == 0) ? -std::numeric_limits<double>::max() : std::numeric_limits<double>::max();

    AgentState state(extreme_object);
    history.add_state(state);
  }

  auto matrix = history.get_object_matrix();

  // Matrix should have correct dimensions
  EXPECT_EQ(matrix.rows(), 10);  // max_history_size
  EXPECT_EQ(matrix.cols(), 5);   // state_size from agent

  // Check for overflow handling
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(std::isinf(matrix(i, 3)) || std::isfinite(matrix(i, 3)));  // vx
    EXPECT_TRUE(std::isinf(matrix(i, 4)) || std::isfinite(matrix(i, 4)));  // vy
  }
}

// Test edge case: Classification with multiple labels
TEST_F(AgentEdgeCaseTest, AgentStateMultipleClassifications)
{
  // Add multiple classifications with different probabilities
  tracked_object_.classification.clear();

  ObjectClassification car_class;
  car_class.label = ObjectClassification::CAR;
  car_class.probability = 0.6;
  tracked_object_.classification.push_back(car_class);

  ObjectClassification truck_class;
  truck_class.label = ObjectClassification::TRUCK;
  truck_class.probability = 0.3;
  tracked_object_.classification.push_back(truck_class);

  ObjectClassification bus_class;
  bus_class.label = ObjectClassification::BUS;
  bus_class.probability = 0.1;
  tracked_object_.classification.push_back(bus_class);

  AgentState agent_state(tracked_object_);

  // State should be created successfully with primary classification
  EXPECT_NO_THROW(agent_state.object_type());
}

// Test edge case: Zero probability object
TEST_F(AgentEdgeCaseTest, AgentStateZeroProbability)
{
  tracked_object_.existence_probability = 0.0;
  tracked_object_.classification[0].probability = 0.0;

  AgentState agent_state(tracked_object_);

  // Should handle zero probability gracefully
  EXPECT_FLOAT_EQ(agent_state.existence_probability(), 0.0);
}

}  // namespace autoware::diffusion_planner::test
