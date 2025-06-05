cmake_minimum_required(VERSION 3.22)
project(autoware_diffusion_planner)

find_package(autoware_cmake REQUIRED)
autoware_package()

# Options
option(CUDA_VERBOSE "Verbose output of CUDA modules" OFF)

# Find CUDA
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if(CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
  if(CUDA_VERBOSE)
    message(STATUS "CUDA is available!")
    message(STATUS "CUDA Libs: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  endif()
  unset(CUDA_cublas_device_LIBRARY CACHE)
  set(CUDA_AVAIL ON)
else()
  message(WARNING "CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif()

# Find TensorRT libraries
option(TRT_AVAIL "TensorRT available" OFF)
find_library(NVINFER nvinfer)
find_library(NVONNXPARSER nvonnxparser)
if(NVINFER AND NVONNXPARSER)
  if(CUDA_VERBOSE)
    message(STATUS "TensorRT is available!")
    message(STATUS "NVINFER: ${NVINFER}")
    message(STATUS "NVONNXPARSER: ${NVONNXPARSER}")
  endif()
  set(TRT_AVAIL ON)
else()
  message(WARNING "TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# Find CUDNN library (optional, often used with TRT)
option(CUDNN_AVAIL "CUDNN available" OFF)
find_library(CUDNN_LIBRARY
  NAMES libcudnn.so
  PATHS $ENV{LD_LIBRARY_PATH} ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  DOC "CUDNN library."
)
if(CUDNN_LIBRARY)
  if(CUDA_VERBOSE)
    message(STATUS "CUDNN is available!")
    message(STATUS "CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  endif()
  set(CUDNN_AVAIL ON)
else()
  message(WARNING "CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()

# ONNX Runtime (your existing)
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

# Add compile option to suppress deprecation warnings (optional)
add_compile_options(-Wno-deprecated-declarations)

# Add your component library
ament_auto_add_library(autoware_diffusion_planner_component SHARED
  src/diffusion_planner_node.cpp
  src/conversion/lanelet.cpp
  src/conversion/agent.cpp
  src/conversion/ego.cpp
  src/postprocessing/postprocessing_utils.cpp
  src/preprocessing/lane_segments.cpp
  src/preprocessing/preprocessing_utils.cpp
  src/preprocessing/traffic_signals.cpp
  src/utils/utils.cpp
  src/utils/marker_utils.cpp
)

# Link libraries for ONNX Runtime, CUDA, TensorRT as available
target_link_libraries(autoware_diffusion_planner_component
  ${ONNXRUNTIME_LIB}
)

if(TRT_AVAIL AND CUDA_AVAIL)
  target_include_directories(autoware_diffusion_planner_component PUBLIC
    ${CUDA_INCLUDE_DIRS}
  )
  target_link_libraries(autoware_diffusion_planner_component
    ${NVINFER}
    ${NVONNXPARSER}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
  )
endif()

if(CUDNN_AVAIL)
  target_link_libraries(autoware_diffusion_planner_component
    ${CUDNN_LIBRARY}
  )
endif()

rclcpp_components_register_node(autoware_diffusion_planner_component
  PLUGIN "autoware::diffusion_planner::DiffusionPlanner"
  EXECUTABLE autoware_diffusion_planner_node
)

if(BUILD_TESTING)
  ament_add_ros_isolated_gtest(test_diffusion_planner
    tests/agent_test.cpp
    tests/lanelet_test.cpp
    tests/ego_test.cpp
    tests/lane_segments_test.cpp
    tests/pre_processing_utils_test.cpp
    tests/fixed_queue_test.cpp
    tests/postprocessing_utils_test.cpp
    tests/utils_test.cpp
    tests/marker_utils_test.cpp)

  target_link_libraries(test_diffusion_planner autoware_diffusion_planner_component)
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
