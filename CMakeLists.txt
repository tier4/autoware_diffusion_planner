cmake_minimum_required(VERSION 3.22)
project(autoware_diffusion_planner)

find_package(autoware_cmake REQUIRED)
autoware_package()

option(ONNXRUNTIME_DIR "Path to ONNX Runtime install root" "$ENV{HOME}/onnxruntime-linux-x64-gpu-1.20.1")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime-linux-x64-gpu-1.20.1")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

include_directories(
  include
  ${ONNXRUNTIME_INCLUDE_DIR}
)

ament_auto_add_library(autoware_diffusion_planner_component SHARED
  src/diffusion_planner_node.cpp
  src/preprocessing/lane_segments.cpp
  src/conversion/lanelet.cpp
  src/conversion/agent.cpp
  src/conversion/ego.cpp
)

target_link_libraries(autoware_diffusion_planner_component
  ${ONNXRUNTIME_LIB}
)


rclcpp_components_register_node(autoware_diffusion_planner_component
  PLUGIN "autoware::diffusion_planner::DiffusionPlanner"
  EXECUTABLE autoware_diffusion_planner_node
)

if(BUILD_TESTING)
  ament_add_ros_isolated_gtest(test_diffusion_planner
  tests/agent_test.cpp)

target_link_libraries(test_diffusion_planner autoware_diffusion_planner_component)

endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
