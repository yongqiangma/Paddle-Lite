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

#include "lite/kernels/vulkan/pool_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

typedef struct {
  int in_dims[4];
  int out_dims[4];
  int ksize[2];
  int strides[2];
  int paddings[2];
  int out_im_dims[2];
} pool_shader_param;

void PoolCompute::PrepareForRun() {
  auto& param = Param<operators::PoolParam>();
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  auto input = param.x;
  auto output = param.output;
  auto ksize = param.ksize;
  auto strides = param.strides;
  auto paddings = param.paddings;

  std::string& pooling_type = param.pooling_type;
  if (param.global_pooling) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings->data()[i] = 0;
      ksize[i] = static_cast<int>(in_dims[i + 2]);
    }
  }
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  bool ceil_mode = param.ceil_mode;
  bool use_quantizer = param.use_quantizer;
  std::string& data_format = param.data_format;

  pool_shader_param shader_param;
  shader_param.in_dims[0] = in_dims[0];
  shader_param.in_dims[1] = in_dims[1];
  shader_param.in_dims[2] = in_dims[2];
  shader_param.in_dims[3] = in_dims[3];

  shader_param.out_dims[0] = out_dims[0];
  shader_param.out_dims[1] = out_dims[1];
  shader_param.out_dims[2] = out_dims[2];
  shader_param.out_dims[3] = out_dims[3];

  shader_param.ksize[0] = ksize[0];
  shader_param.ksize[1] = ksize[1];

  shader_param.strides[0] = strides[0];
  shader_param.strides[1] = strides[1];

  shader_param.paddings[0] = paddings->data()[0];
  shader_param.paddings[1] = paddings->data()[1];

  DDim out_im_dims = InitImageDimInfoWith(out_dims);
  shader_param.out_im_dims[0] = out_im_dims[0];
  shader_param.out_im_dims[1] = out_im_dims[1];

  auto& ctx = this->ctx_->template As<VULKANContext>();

  device = ctx.device();
  lite::vulkan::VulkanMem* imagedataout =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(
          TARGET(kVULKAN), out_im_dims[0], out_im_dims[1]);

  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          sizeof(pool_shader_param));
  CopySync<TARGET(kVULKAN)>(
      buf_param, &shader_param, sizeof(pool_shader_param), IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);

  if (pooling_type == "max") {
    device->CreateComputePipelines("pool2d_max");
  } else if (pooling_type == "avg") {
    device->CreateComputePipelines("pool2d_avg");
  }

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

  device->AllocateCommandBuffers(&cmd);
  device->BeginCommandBuffer(cmd, 0);
  VLOG(5) << "vkCmdBindPipeline !!!!\n";
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipeline);

  // Bind descriptor set.
  vkCmdBindDescriptorSets(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          device->pipelineLayout,
                          0,
                          1,
                          device->descriptorSet,
                          0,
                          nullptr);

  vkCmdDispatch(cmd, (out_im_dims[0] + 15) / 16, (out_im_dims[1] + 15) / 16, 1);
  device->EndCommandBuffer(cmd);
}
void PoolCompute::Run() { device->cmds.emplace_back(cmd); }
}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pool2d,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::PoolCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .Finalize();
