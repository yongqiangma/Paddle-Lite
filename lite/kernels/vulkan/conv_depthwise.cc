// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#include "lite/kernels/vulkan/conv_depthwise.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

typedef struct {
  int input[4];
  int filter[4];
  int out[4];
  int strides[2];
  int pads[2];
  int dilations[2];
  int out_im[2];
  bool flag[2];
} conv_shader_param;

void DepthwiseConv::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<VULKANContext>();
  auto input = param.x;
  auto filter = param.filter;
  auto bias = param.bias;
  auto fuse_relu = param.fuse_relu;
  auto output = param.output;
  auto stride = param.strides;
  auto pad = param.paddings;
  auto dilation = param.dilations;
  const auto b_data = param.bias ? param.bias->data<float>() : nullptr;

  lite::DDim input_dim = input->dims();
  lite::DDim output_dim = output->dims();
  lite::DDim filter_dim = filter->dims();
  std::vector<int> input_dim_v;
  for (int i = 0; i < input_dim.size(); i++) {
    input_dim_v.push_back(input_dim.data()[i]);
  }

  conv_shader_param shader_param;
  shader_param.input[0] = input_dim_v.data()[0];
  shader_param.input[1] = input_dim_v.data()[1];
  shader_param.input[2] = input_dim_v.data()[2];
  shader_param.input[3] = input_dim_v.data()[3];

  shader_param.out[0] = output_dim.data()[0];
  shader_param.out[1] = output_dim.data()[1];
  shader_param.out[2] = output_dim.data()[2];
  shader_param.out[3] = output_dim.data()[3];

  shader_param.filter[0] = filter_dim.data()[0];
  shader_param.filter[1] = filter_dim.data()[1];
  shader_param.filter[2] = filter_dim.data()[2];
  shader_param.filter[3] = filter_dim.data()[3];

  shader_param.strides[0] = stride[0];
  shader_param.strides[1] = stride[1];

  shader_param.dilations[0] = dilation->data()[0];
  shader_param.dilations[1] = dilation->data()[1];

  shader_param.pads[0] = pad->data()[0];
  shader_param.pads[1] = pad->data()[1];

  shader_param.flag[0] = bias;
  shader_param.flag[1] = fuse_relu;

  DDim out_image_dim = InitImageDimInfoWith(output_dim);
  shader_param.out_im[0] = out_image_dim[0];
  shader_param.out_im[1] = out_image_dim[1];

  device = ctx.device();
  lite::vulkan::VulkanMem* imagedataout =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(
          TARGET(kVULKAN), out_image_dim[0], out_image_dim[1]);

  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          sizeof(conv_shader_param));
  CopySync<TARGET(kVULKAN)>(
      buf_param, &shader_param, sizeof(conv_shader_param), IoDirection::HtoD);

  lite::Tensor filter_im;

  filter->Resize(lite::DDim{std::vector<int64_t>({filter_dim.data()[1],
                                                  filter_dim.data()[0],
                                                  filter_dim.data()[2],
                                                  filter_dim.data()[3]})});

  VulkanBuf2Image(device, filter, &filter_im, BufTYPE::NCHW);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);
  std::string conv = "conv";
  if (fuse_relu) {
    conv += "_RELU";
  }

  device->CreateComputePipelines(conv);

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);

  int input_index = 0;
  const lite::vulkan::VulkanMem* bufdata =
      input->data<lite::vulkan::VulkanMem>();
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bufdata,
                               input_index,
                               types[input_index],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  int output_index = 1;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               imagedataout,
                               output_index,
                               types[output_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  int param_index = 2;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               buf_param,
                               param_index,
                               types[param_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  const lite::vulkan::VulkanMem* filterim =
      filter_im.data<lite::vulkan::VulkanMem>();
  int filter_index = 3;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               filterim,
                               filter_index,
                               types[filter_index],
                               VK_IMAGE_LAYOUT_GENERAL);
  int bias_index = 4;
  const lite::vulkan::VulkanMem* bias_param =
      bias->data<lite::vulkan::VulkanMem>();

  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bias_param,
                               bias_index,
                               types[bias_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  device->AllocateCommandBuffers(&cmd);
  device->BeginCommandBuffer(cmd, 0);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipeline);

  vkCmdBindDescriptorSets(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          device->pipelineLayout,
                          0,
                          1,
                          device->descriptorSet,
                          0,
                          nullptr);

  VLOG(5) << "vkCmdBindDescriptorSets!!!!";

  vkCmdDispatch(
      cmd, (out_image_dim[0] + 15) / 16, (out_image_dim[1] + 15) / 16, 1);
  device->EndCommandBuffer(cmd);
}
void DepthwiseConv::Run() { device->cmds.emplace_back(cmd); }

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
