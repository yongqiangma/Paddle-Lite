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

#include <map>
#include <string>
#include <utility>

#include "lite/backends/vulkan/shader_hex.h"
#include "lite/utils/io.h"

class ShaderMaps {
 public:
  ShaderMaps();

  std::pair<const uint32_t*, uint32_t> find(const char* key) {
    auto iter = shader_maps.find(key);
    if (iter != shader_maps.end()) {
      return iter->second;
    }
    LOG(FATAL) << "The vulkan shader :" << key << "not support";
    return std::make_pair(nullptr, 0);
  }
  ~ShaderMaps() {}

 private:
  std::map<const char*, std::pair<const uint32_t*, uint32_t>> shader_maps;
};
