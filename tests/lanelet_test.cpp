#include "lanelet_test.hpp"

namespace autoware::diffusion_planner::test
{

TEST_F(LaneletTest, ConvertToLaneSegments)
{
  LaneletConverter converter(lanelet_map_, 10, 20, 1.0);

  auto lane_segments = converter.convert_to_lane_segments(10);

  EXPECT_EQ(lane_segments.size(), 1);               // Expect one lanelet to be converted
  EXPECT_EQ(lane_segments[0].polyline.size(), 10);  // Expect 10 points in the polyline
}

TEST_F(LaneletTest, ConvertLaneletToPolyline)
{
  LaneletConverter converter(lanelet_map_, 10, 20, 1.0);

  geometry_msgs::msg::Point position;
  position.x = 5.0;
  position.y = 0.0;

  auto polyline_data = converter.convert(position, 15.0);

  ASSERT_TRUE(polyline_data.has_value());
}

TEST_F(LaneletTest, FromLineString)
{
  LaneletConverter converter(lanelet_map_, 10, 20, 1.0);

  auto points = converter.from_linestring(centerline_);

  EXPECT_EQ(points.size(), 3);  // Expect 3 points in the centerline
  EXPECT_FLOAT_EQ(points[0].x(), 0.0);
  EXPECT_FLOAT_EQ(points[1].x(), 10.0);
  EXPECT_FLOAT_EQ(points[2].x(), 20.0);
}

TEST_F(LaneletTest, InterpolateLane)
{
  std::vector<LanePoint> waypoints = {
    LanePoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    LanePoint(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    LanePoint(20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  };

  auto interpolated = interpolate_lane(waypoints, 5.0);

  EXPECT_EQ(interpolated.size(), 5);  // Expect 5 interpolated points
  EXPECT_FLOAT_EQ(interpolated[0].x(), 0.0);
  EXPECT_FLOAT_EQ(interpolated[4].x(), 20.0);
}

}  // namespace autoware::diffusion_planner::test
