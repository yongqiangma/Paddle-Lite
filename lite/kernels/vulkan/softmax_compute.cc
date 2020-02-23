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

#include "lite/kernels/vulkan/softmax_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

typedef struct {
  int compute_size;
  int axis_size;
  int outer_num;
  int inner_num;
} softmax_shader_param;

void SoftmaxCompute::PrepareForRun() {
  param = Param<operators::SoftmaxParam>();
  auto input = param.x;
  auto out = param.output;

  auto x_dims = param.x->dims();
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int axis_size = x_dims[axis];
  int compute_size = outer_num * inner_num;

  auto& ctx = this->ctx_->template As<VULKANContext>();
  device = ctx.device();

  lite::Tensor tempTensor;
  softmax_shader_param shader_param;
  shader_param.compute_size = compute_size;
  shader_param.axis_size = axis_size;
  shader_param.outer_num = outer_num;
  shader_param.inner_num = inner_num;
  lite::vulkan::VulkanMem* out_mem =
      (lite::vulkan::VulkanMem*)out->mutable_data<float>(TARGET(kVULKAN));

  tempTensor.Resize(out->dims());
  lite::vulkan::VulkanMem* temp_mem =
      (lite::vulkan::VulkanMem*)tempTensor.mutable_data<float>(TARGET(kVULKAN));

  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          sizeof(softmax_shader_param));
  CopySync<TARGET(kVULKAN)>(buf_param,
                            &shader_param,
                            sizeof(softmax_shader_param),
                            IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);
  std::string shader = "softmax";

  device->CreateComputePipelines(shader);

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);

  const lite::vulkan::VulkanMem* in_mem =
      input->data<lite::vulkan::VulkanMem>();
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               in_mem,
                               0,
                               types[0],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  device->UpdateDescriptorSets(
      device->GetVkDevice(), out_mem, 1, types[1], VK_IMAGE_LAYOUT_GENERAL);

  device->UpdateDescriptorSets(
      device->GetVkDevice(), buf_param, 2, types[2], VK_IMAGE_LAYOUT_GENERAL);

  device->UpdateDescriptorSets(
      device->GetVkDevice(), temp_mem, 3, types[3], VK_IMAGE_LAYOUT_GENERAL);
  // device->AllocateCommandBuffers(&cmd);
  device->AllocateCommandBuffers();
  device->BeginCommandBuffer(device->cmd, 0);
  VLOG(5) << "vkCmdBindPipeline !!!!\n";
  vkCmdBindPipeline(
      device->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipeline);

  // Bind descriptor set.
  vkCmdBindDescriptorSets(device->cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          device->pipelineLayout,
                          0,
                          1,
                          device->descriptorSet,
                          0,
                          nullptr);

  vkCmdDispatch(device->cmd, (outer_num + 15) / 16, (inner_num + 15) / 16, 1);
  device->EndCommandBuffer(device->cmd);
  cmds.emplace_back(device->cmd);
  LOG(INFO) << "vkCmdBindDescriptorSets!!!!:" << cmds.size();
}
void SoftmaxCompute::Run() {
  for (int i = 0; i < cmds.size(); i++) {
    device->cmds.emplace_back(cmds[i]);
  }
}
}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(softmax,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::SoftmaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .Finalize();
