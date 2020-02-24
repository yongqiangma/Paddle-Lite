if (NOT LITE_WITH_VULKAN)
    return()
endif()

    lite_cc_library(vulkan_device_info SRCS vk_device.cc vk_command.cc )
    #lite_cc_library(vulkan_shader SRCS shader_map.cc)
    lite_cc_library(vulkan_target_wrapper SRCS target_wrapper.cc ../../core/device_info.cc DEPS vulkan_device_info)

    set(vulkan_kernels
        vulkan_device_info
    )
    MESSAGE(STATUS "-===============================")

    set(vulkan_kernels "${vulkan_kernels}" CACHE INTERNAL "vulkan kernels")

    

    add_custom_command(
    COMMAND python ${CMAKE_SOURCE_DIR}/lite/backends/vulkan/build_shaders.py
    ${CMAKE_SOURCE_DIR}/lite/backends/vulkan
    OUTPUT outputshader.h # not a real path to the output to force it execute every time.

    )


    #add_dependencies(vulkan_device_info  run)
    add_custom_target(shader  ALL DEPENDS outputshader.h )
    add_dependencies(vulkan_device_info shader)
