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

#include "lite/kernels/vulkan/fc_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

typedef struct { int dims[3]; } fc_shader_param;

void FcCompute::PrepareForRun() {
  param = Param<operators::FcParam>();
  ReInitWhenNeeded();
  auto input = param.input;
  auto weight = param.w;
  auto bias = param.bias;
  auto out = param.output;

  auto x_dims = param.input->dims();
  auto& w_dims = param.w->dims();
  auto& out_dims = param.output->dims();

  fc_shader_param shader_param;
  shader_param.dims[0] = m_;
  shader_param.dims[1] = n_;
  shader_param.dims[2] = k_;

  auto& ctx = this->ctx_->template As<VULKANContext>();
  device = ctx.device();

  lite::Tensor weight_trans, input_buf, inbuf_cpu;
  if (!flag_trans_weights_) {
    flag_trans_weights_ = true;
    VulkanMatrixTrans(device, weight, &weight_trans);
  }
  CHECK_EQ(4, static_cast<int>(x_dims.size()));
  // VulkanPrintImage(device, input,BufTYPE::NCHW);
  VulkanImage2Buf(device, input, &input_buf, BufTYPE::NCHW);

  cmds.emplace_back(device->cmd);
  /*
    CopyToHost(device, &input_buf,&inbuf_cpu);

    auto* inbuf_cpu_d = inbuf_cpu.data<float>();

    for (int i = 0; i < x_dims.production(); i++) {
    LOG(INFO)<<"input buf data:"<<i<<" : "<<inbuf_cpu_d[i];
    }
    */

  lite::vulkan::VulkanMem* out_mem =
      (lite::vulkan::VulkanMem*)out->mutable_data<float>(TARGET(kVULKAN));

  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          sizeof(fc_shader_param));
  CopySync<TARGET(kVULKAN)>(
      buf_param, &shader_param, sizeof(fc_shader_param), IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);
  std::string fc = "gemmbuf";
  if (param.bias) {
    // fc += "_BIAS";
  }

  device->CreateComputePipelines(fc);

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);

  const lite::vulkan::VulkanMem* in_mem =
      input_buf.data<lite::vulkan::VulkanMem>();
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               in_mem,
                               0,
                               types[0],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  const lite::vulkan::VulkanMem* w_mem =
      weight_trans.data<lite::vulkan::VulkanMem>();

  device->UpdateDescriptorSets(device->GetVkDevice(),
                               w_mem,
                               1,
                               types[1],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  const lite::vulkan::VulkanMem* b_mem = bias->data<lite::vulkan::VulkanMem>();
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               b_mem,
                               2,
                               types[2],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  device->UpdateDescriptorSets(
      device->GetVkDevice(), out_mem, 3, types[3], VK_IMAGE_LAYOUT_GENERAL);

  device->UpdateDescriptorSets(
      device->GetVkDevice(), buf_param, 4, types[4], VK_IMAGE_LAYOUT_GENERAL);

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

  vkCmdDispatch(device->cmd, (1 + 15) / 16, (n_ + 15) / 16, 1);
  device->EndCommandBuffer(device->cmd);
  cmds.emplace_back(device->cmd);
}
void FcCompute::Run() {
  for (int i = 0; i < cmds.size(); i++) {
    device->cmds.emplace_back(cmds[i]);
  }
}
}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    fc, kVULKAN, kFloat, kNCHW, paddle::lite::kernels::vulkan::FcCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .Finalize();
