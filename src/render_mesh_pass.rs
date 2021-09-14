use glam::*;
use shared::MeshShaderConstants;
use std::any::TypeId;
use std::default::Default;
use std::ffi::CString;

use ash::{vk, Device};

use crate::game::SpvFile;
use crate::mesh::*;
use crate::render_pass_common::*;
use crate::vulkan_base::*;
use crate::vulkan_helpers::*;

#[allow(dead_code)]
pub struct RenderMeshPass {
    pub pipeline_layout: vk::PipelineLayout,
    pub index_buffer: Option<VkBuffer>,
    pub index_buffer_gpu: Option<VkBuffer>,
    pub vertex_buffer: VkBuffer,
    pub vertex_buffer_gpu: VkBuffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub shader_module: vk::ShaderModule,
    pub object_to_world: Mat4,
}

impl RenderMeshPass {
    pub fn new<VertexType: IVertex>(
        device: &Device,
        allocator: &vk_mem::Allocator,
        descriptor_pool: &vk::DescriptorPool,
        render_pass: &vk::RenderPass,
        view_scissor: &VkViewScissor,
        mesh: &Mesh<VertexType>,
        shader_file: &SpvFile,
    ) -> RenderMeshPass
    where
        VertexType: Copy + 'static,
    {
        let alloc_info_cpu = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuOnly,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let alloc_info_gpu = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let submesh = &mesh.submeshes[0];

        let vertex_buffer_data = &submesh.vertex_buffer.vertices;

        let vertex_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&vertex_buffer_data[..]) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vertex_buffer = VkBuffer::new(allocator, &vertex_buffer_info, &alloc_info_cpu);
        vertex_buffer.copy_from_slice(&vertex_buffer_data[..], 0);

        let vertex_buffer_gpu_info = vk::BufferCreateInfo {
            size: std::mem::size_of_val(&vertex_buffer_data[..]) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let vertex_buffer_gpu = VkBuffer::new(allocator, &vertex_buffer_gpu_info, &alloc_info_gpu);

        let index_count = if let Some(index_buffer) = &submesh.index_buffer {
            index_buffer.len() as u32
        } else {
            0
        };

        let (index_buffer, index_buffer_gpu) =
            if let Some(index_buffer_data) = &submesh.index_buffer {
                let index_buffer_info = vk::BufferCreateInfo {
                    size: std::mem::size_of_val(&index_buffer_data[..]) as u64,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                };

                let index_buffer = VkBuffer::new(allocator, &index_buffer_info, &alloc_info_cpu);
                index_buffer.copy_from_slice(&index_buffer_data[..], 0);

                let index_buffer_gpu_info = vk::BufferCreateInfo {
                    size: std::mem::size_of_val(&index_buffer_data[..]) as u64,
                    usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                };

                let index_buffer_gpu =
                    VkBuffer::new(allocator, &index_buffer_gpu_info, &alloc_info_gpu);

                (Some(index_buffer), Some(index_buffer_gpu))
            } else {
                (None, None)
            };

        let desc_layout_bindings = [
            //vk::DescriptorSetLayoutBinding {
            //binding: 0,
            //descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            //descriptor_count: 1,
            //stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
            //..Default::default()
            //}
        ];
        let descriptor_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

        let desc_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_info, None) }.unwrap();

        let desc_set_layouts = &[desc_set_layout];

        let descriptor_sets = {
            let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(*descriptor_pool)
                .set_layouts(desc_set_layouts);

            unsafe { device.allocate_descriptor_sets(&desc_alloc_info) }.unwrap()
        };

        let write_desc_sets = [
            //vk::WriteDescriptorSet {
            //dst_set: descriptor_sets[0],
            //dst_binding: 0,
            //descriptor_count: 1,
            //descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            //p_buffer_info: &uniform_buffer_descriptor,
            //..Default::default()
            //}
        ];
        unsafe { device.update_descriptor_sets(&write_desc_sets, &[]) };

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<MeshShaderConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::all())
            .build();

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(desc_set_layouts)
            .push_constant_ranges(&[push_constant_range])
            .build();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&layout_create_info, None) }.unwrap();

        let entry_points = vec![(
            ShaderEntryPoint {
                entry_point: String::from("main_vs"),
            },
            ShaderEntryPoint {
                entry_point: String::from("main_fs"),
            },
        )];

        let shader_module: vk::ShaderModule = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(shader_file.data.as_slice()),
                    None,
                )
                .expect("Shader module creation error")
        };

        let shader_module_sets = entry_points
            .into_iter()
            .map(|(vert, frag)| {
                let vert_module = shader_module;
                let vert_entry_point = CString::new(vert.entry_point).unwrap();
                let frag_module = shader_module;
                let frag_entry_point = CString::new(frag.entry_point).unwrap();
                ShaderModules::vert_frag(
                    vert_module,
                    vert_entry_point,
                    frag_module,
                    frag_entry_point,
                )
            })
            .collect::<Vec<_>>();
        let shader_stage_create_infos = shader_module_sets
            .iter()
            .map(|modules| {
                [
                    vk::PipelineShaderStageCreateInfo {
                        module: modules.vertex_module.unwrap(),
                        p_name: (*modules.vertex_entry_point.as_ref().unwrap()).as_ptr(),
                        stage: vk::ShaderStageFlags::VERTEX,
                        ..Default::default()
                    },
                    vk::PipelineShaderStageCreateInfo {
                        module: modules.fragment_module.unwrap(),
                        p_name: (*(modules.fragment_entry_point.as_ref().unwrap())).as_ptr(),
                        stage: vk::ShaderStageFlags::FRAGMENT,
                        ..Default::default()
                    },
                ]
            })
            .collect::<Vec<_>>();

        let offset = 0;

        let vertex_type_id = TypeId::of::<VertexType>();
        let vertex_attribute_descriptions = if vertex_type_id == TypeId::of::<Vertex>() {
            [
                // Position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(offset)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .build(),
                // Normal
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .build(),
                // Texcoord
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // Colour
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(3)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .build(),
            ]
        } else if vertex_type_id == TypeId::of::<AnimatedVertex>() {
            [
                // Position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(offset)
                    .location(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .build(),
                // Normal
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(1)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .build(),
                // Texcoord
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(2)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // Colour
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(3)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .build(),
                // TODO:
            ]
        } else {
            panic!("Unhandled vertex type")
        };

        let mesh_stride = std::mem::size_of::<f32>() as u32 * 3 + // pos
            std::mem::size_of::<f32>() as u32 * 3 + // normal
            std::mem::size_of::<f32>() as u32 * 2 + // texcoord
            std::mem::size_of::<f32>() as u32 * 4; // colour
        let vertex_binding_descriptions = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(mesh_stride)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_attribute_descriptions)
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .build();

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let scissors = &[view_scissor.scissor];
        let viewports = &[view_scissor.viewport];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .scissors(scissors)
            .viewports(viewports);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            ..Default::default()
        };

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            front: noop_stencil_state,
            back: noop_stencil_state,
            max_depth_bounds: 1.0,
            ..Default::default()
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::all(),
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stage_create_infos[0])
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass);

        let graphics_pipelines = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphics_pipeline_info.build()],
                None,
            )
        }
        .unwrap();

        let graphics_pipeline = graphics_pipelines[0];

        RenderMeshPass {
            pipeline_layout,
            index_buffer,
            index_buffer_gpu,
            vertex_buffer,
            vertex_buffer_gpu,
            vertex_count: submesh.vertex_buffer.vertex_count,
            index_count,
            desc_set_layout,
            graphics_pipeline,
            descriptor_sets,
            shader_module,
            object_to_world: Mat4::IDENTITY,
        }
    }

    pub fn update(&self) {}

    pub fn gpu_setup(&self, device: &Device, command_buffer: &vk::CommandBuffer) {
        let vertex_buffer_copy_regions = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: self.vertex_buffer.size,
        };

        let vertex_buffer_barrier = vk::BufferMemoryBarrier {
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            buffer: self.vertex_buffer_gpu.buffer,
            offset: 0,
            size: vertex_buffer_copy_regions.size,
            ..Default::default()
        };

        let vertex_buffer_barrier_end = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            buffer: self.vertex_buffer_gpu.buffer,
            offset: 0,
            size: vertex_buffer_copy_regions.size,
            ..Default::default()
        };

        let mut buffer_memory_barriers = Vec::new();
        let mut buffer_memory_end_barriers = Vec::new();

        buffer_memory_barriers.push(vertex_buffer_barrier);
        buffer_memory_end_barriers.push(vertex_buffer_barrier_end);

        if let (Some(index_buffer), Some(index_buffer_gpu)) =
            (&self.index_buffer, &self.index_buffer_gpu)
        {
            let index_buffer_copy_regions = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: index_buffer.size,
            };
            let index_buffer_barrier = vk::BufferMemoryBarrier {
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                buffer: index_buffer_gpu.buffer,
                offset: 0,
                size: index_buffer_copy_regions.size,
                ..Default::default()
            };
            let index_buffer_barrier_end = vk::BufferMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::INDEX_READ,
                buffer: index_buffer_gpu.buffer,
                offset: 0,
                size: index_buffer_copy_regions.size,
                ..Default::default()
            };

            buffer_memory_barriers.push(index_buffer_barrier);
            buffer_memory_end_barriers.push(index_buffer_barrier_end);

            unsafe {
                device.cmd_copy_buffer(
                    *command_buffer,
                    index_buffer.buffer,
                    index_buffer_gpu.buffer,
                    &[index_buffer_copy_regions],
                );
            }
        }

        unsafe {
            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                buffer_memory_barriers.as_slice(),
                &[],
            );
        }

        unsafe {
            device.cmd_copy_buffer(
                *command_buffer,
                self.vertex_buffer.buffer,
                self.vertex_buffer_gpu.buffer,
                &[vertex_buffer_copy_regions],
            );

            device.cmd_pipeline_barrier(
                *command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                &[],
                buffer_memory_end_barriers.as_slice(),
                &[],
            );
        };
    }

    pub fn draw_main_render_pass(
        &self,
        shader_constants: &shared::MeshShaderConstants,
        device: &Device,
        command_buffer: &vk::CommandBuffer,
    ) {
        unsafe {
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptor_sets[..],
                &[],
            );

            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            device.cmd_bind_vertex_buffers(
                *command_buffer,
                0,
                &[self.vertex_buffer_gpu.buffer],
                &[0],
            );

            if let Some(index_buffer_gpu) = &self.index_buffer_gpu {
                device.cmd_bind_index_buffer(
                    *command_buffer,
                    index_buffer_gpu.buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            let push_constants = shader_constants;
            device.cmd_push_constants(
                *command_buffer,
                self.pipeline_layout,
                ash::vk::ShaderStageFlags::all(),
                0,
                any_as_u8_slice(push_constants),
            );

            if self.index_count > 0 {
                device.cmd_draw_indexed(*command_buffer, self.index_count, 1, 0, 0, 0);
            } else {
                device.cmd_draw(*command_buffer, self.vertex_count, 1, 0, 0);
            }
        }
    }

    pub fn destroy(&self, device: &Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_shader_module(self.shader_module, None);
            if let Some(index_buffer) = &self.index_buffer {
                index_buffer.destroy(allocator);
            }
            if let Some(index_buffer_gpu) = &self.index_buffer_gpu {
                index_buffer_gpu.destroy(allocator);
            }
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
