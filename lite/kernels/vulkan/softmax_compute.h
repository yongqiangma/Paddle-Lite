// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "lite/backends/vulkan/vk_buffer.h"
#include "lite/backends/vulkan/vk_device.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/vulkan/vulkan_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

class SoftmaxCompute : public KernelLite<TARGET(kVULKAN), PRECISION(kFloat)> {
 public:
  void PrepareForRun() override;
  void Run() override;

  virtual ~SoftmaxCompute() = default;

 private:
  operators::SoftmaxParam param;
  std::vector<VkCommandBuffer> cmds;
  std::shared_ptr<lite::vulkan::VulkanDevice> device;
};

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
