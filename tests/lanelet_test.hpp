#ifndef LANELET_TEST_HPP_
#define LANELET_TEST_HPP_

#include "autoware/diffusion_planner/conversion/lanelet.hpp"
#include "gtest/gtest.h"

#include <geometry_msgs/msg/point.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>

#include <memory>
#include <vector>

namespace autoware::diffusion_planner::test
{

class LaneletTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Create a mock lanelet map
    lanelet_map_ = std::make_shared<lanelet::LaneletMap>();

    // Add a lanelet with a centerline
    centerline_ = lanelet::LineString3d(
      lanelet::utils::getId(), {
                                 lanelet::Point3d(lanelet::utils::getId(), 0.0, 0.0, 0.0),
                                 lanelet::Point3d(lanelet::utils::getId(), 10.0, 0.0, 0.0),
                                 lanelet::Point3d(lanelet::utils::getId(), 20.0, 0.0, 0.0),
                               });
    lanelet_ = lanelet::Lanelet(lanelet::utils::getId(), centerline_, centerline_);
    lanelet_.setAttribute("subtype", "road");
    lanelet_map_->add(lanelet_);
  }

  lanelet::LaneletMapPtr lanelet_map_;
  lanelet::LineString3d centerline_;
  lanelet::Lanelet lanelet_;
};

}  // namespace autoware::diffusion_planner::test

#endif  // LANELET_TEST_HPP_
