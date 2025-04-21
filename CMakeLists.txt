cmake_minimum_required(VERSION 3.22)
project(autoware_diffusion_planner)

find_package(autoware_cmake REQUIRED)
autoware_package()

# control validator
ament_auto_add_library(autoware_diffusion_planner_component SHARED
  src/diffusion_planner_node.cpp
)
target_link_libraries(autoware_diffusion_planner_component)
rclcpp_components_register_node(autoware_diffusion_planner_component
  PLUGIN "autoware::diffusion_planner::DiffusionPlanner"
  EXECUTABLE autoware_diffusion_planner_node
)

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   ament_lint_auto_find_test_dependencies()

#   file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)
#   ament_add_ros_isolated_gtest(test_autoware_diffusion_planner

#     ${TEST_SOURCES}
#   )
#   target_link_libraries(test_autoware_diffusion_planner
#     autoware_diffusion_planner_component
#   )
# endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
