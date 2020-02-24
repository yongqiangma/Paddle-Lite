# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import os
import logging
import sys

path = sys.argv[1]
micro_shader_dict = {'conv_direct':['_RELU','_RELU6'],
                    'conv':['_RELU','_RELU6'],
                    'gemm':['_BIAS']}
shader_hex_file = path+"/shader_hex.h"
shader_header_file = path+"/shader_header.h"
shader_map_file = path+"/shader_map.cc"
in_file = "relu.comp"
out_file = "temp.o"
command = "find " + path + "/vk_shader -name \"*.comp\""
compiler = "/code/vulkan/vulkan1.1.92.1/x86_64/bin/glslangValidator"
# compiler = "glslangValidator"


comp_files = os.popen(command).read().split('\n')

print comp_files
print comp_files[0]
print comp_files[1]
# comp_files[0] = comp_files[1]
data = "#pragma once\n\n"
header = "#pragma once\n\n"
header += "struct vulkan_shader {\n"
header +=  "  const char* name;\n"
header +=  "  const uint32_t* data;\n"
header +=  "  uint32_t size;\n"
header +=  "  };\n\n"
header +=  "static const vulkan_shader vulkan_shaders[] = {\n"

shader_map = "#include \"shader_map.h\"\n\n"

shader_map += "ShaderMaps::ShaderMaps() {\n"
#print comp_files[0]
for file_ in comp_files:
    if file_ == '':
        continue
    print file_.split('/')[-1]
    
    in_file = file_.split('/')[-1]
    out_file = file_+".out"
    file_name = in_file.split('.')
    print file_name[0]
    list_micros =[]
    list_micros = micro_shader_dict.get(file_name[0],'')
    len(list_micros)


    cnt = len(list_micros)+1
    comp_name = file_name[0]
    compile_param =''
    for i in range(cnt):

        name_temp = comp_name +"_hex_data"
        command = compiler + " -V "+ file_ + compile_param +" -x -o hex.spv"
        print command
        # subprocess.Popen(command, shell=True)
        print os.popen(command).read()

        data += "static const uint32_t " + name_temp+"[] = { \n"
        data += open('hex.spv','rb').read()
        #print open('hex.spv','rb').read()
        data += "};\n"
        
        header += "  {\""+comp_name +"\", " + name_temp +", sizeof(" + name_temp + ")},\n"

        shader_map += "    shader_maps.insert( std::make_pair(\""+ comp_name + "\", std::make_pair(" + name_temp +", sizeof(" + name_temp + "))));\n"
        if i < cnt-1:
            comp_name = file_name[0] + list_micros[i]
            compile_param = " -D"+list_micros[i]

with open(shader_hex_file, 'w') as f:
    logging.info("write vulkan shader hex data to %s" % shader_hex_file)

    f.write(data)
    f.close()
with open(shader_header_file, 'w') as f:
    f.write(header)
    f.write("};")
    f.close()

with open(shader_map_file, 'w') as f:
    f.write(shader_map)
    f.write("\n}")
    f.close()
