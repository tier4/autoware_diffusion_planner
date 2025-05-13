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

#include "autoware/diffusion_planner/posprocessing/posprocessing_utils.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

namespace autoware::diffusion_planner::posprocessing
{
void transform_output_matrix(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, long column_idx,
  long row_idx, bool do_translation)
{
  Eigen::Matrix<float, 4, OUTPUT_T> xy_block = Eigen::Matrix<float, 4, OUTPUT_T>::Zero();
  xy_block.block<2, OUTPUT_T>(0, 0) =
    output_matrix.block<2, OUTPUT_T>(row_idx, column_idx * OUTPUT_T);
  xy_block.row(3) = do_translation ? Eigen::Matrix<float, 1, OUTPUT_T>::Ones()
                                   : Eigen::Matrix<float, 1, OUTPUT_T>::Zero();

  Eigen::Matrix<float, 4, OUTPUT_T> transformed_block = transform_matrix * xy_block;
  output_matrix.block<2, OUTPUT_T>(row_idx, column_idx * OUTPUT_T) =
    transformed_block.block<2, OUTPUT_T>(0, 0);
};

}  // namespace autoware::diffusion_planner::posprocessing
