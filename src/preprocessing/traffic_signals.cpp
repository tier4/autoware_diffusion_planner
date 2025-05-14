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

#include "autoware/diffusion_planner/preprocessing/traffic_signals.hpp"

namespace autoware::diffusion_planner::preprocess
{
void process_traffic_signals(
  const autoware_perception_msgs::msg::TrafficLightGroupArray::ConstSharedPtr msg,
  std::map<lanelet::Id, TrafficSignalStamped> & traffic_signal_id_map)
{
  // clear previous observation
  traffic_signal_id_map.clear();

  if (!msg) {
    return;
  }
  for (const auto & signal : msg->traffic_light_groups) {
    TrafficSignalStamped traffic_signal;
    traffic_signal.stamp = msg->stamp;
    traffic_signal.signal = signal;
    traffic_signal_id_map[signal.traffic_light_group_id] = traffic_signal;

    // TODO (Daniel): implement fallback for unknown signals
    // const bool is_unknown_observation =
    //   std::any_of(signal.elements.begin(), signal.elements.end(), [](const auto & element) {
    //     return element.color == autoware_perception_msgs::msg::TrafficLightElement::UNKNOWN;
    //   });
  }
}

}  // namespace autoware::diffusion_planner::preprocess
