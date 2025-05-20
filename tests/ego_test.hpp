#ifndef EGO_TEST_HPP_
#define EGO_TEST_HPP_

#include "autoware/diffusion_planner/conversion/ego.hpp"
#include "gtest/gtest.h"

#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace autoware::diffusion_planner::test
{

class EgoTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Initialize mock odometry message
    odometry_msg_.pose.pose.position.x = 10.0;
    odometry_msg_.pose.pose.position.y = 5.0;
    odometry_msg_.twist.twist.linear.x = 3.0;
    odometry_msg_.twist.twist.linear.y = 0.0;
    odometry_msg_.twist.twist.angular.z = 0.1;

    // Initialize mock acceleration message
    acceleration_msg_.accel.accel.linear.x = 0.5;
    acceleration_msg_.accel.accel.linear.y = 0.2;

    wheel_base_ = 2.5f;  // Example wheelbase
  }

  nav_msgs::msg::Odometry odometry_msg_;
  geometry_msgs::msg::AccelWithCovarianceStamped acceleration_msg_;
  float wheel_base_;
};

}  // namespace autoware::diffusion_planner::test

#endif  // EGO_TEST_HPP_
