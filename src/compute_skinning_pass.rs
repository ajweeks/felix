use glam::*;
use shared::ComputeSkinningShaderConstants;
use std::default::Default;
use std::ffi::CString;

use ash::{vk, Device};

use crate::game::SpvFile;
use crate::mesh::*;
use crate::render_pass_common::*;
use crate::vulkan_base::*;
use crate::vulkan_helpers::*;

#[allow(dead_code)]
pub struct ComputeSkinningPass {
    pub pipeline_layout: vk::PipelineLayout,
    pub bone_pose_buffer: VkBuffer,
    pub desc_set_layout: vk::DescriptorSetLayout,
    pub compute_pipeline: vk::Pipeline,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub shader_module: vk::ShaderModule,
    pub object_to_world: Mat4,
}

impl ComputeSkinningPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        allocator: &vk_mem::Allocator,
        descriptor_pool: &vk::DescriptorPool,
        mesh: &Mesh,
        shader_file: &SpvFile,
    ) -> ComputeSkinningPass {
        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let bone_pose_data = &mesh.bone_poses;

        let bone_pose_buffer_info = vk::BufferCreateInfo {
            size: std::mem::size_of::<shared::BonePoseBuffer>() as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let bone_pose_buffer = VkBuffer::new(allocator, &bone_pose_buffer_info, &alloc_info);
        bone_pose_buffer.copy_from_slice(&bone_pose_data[..], 0);

        let desc_layout_bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        }];
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

        let write_desc_sets = [];
        unsafe { device.update_descriptor_sets(&write_desc_sets, &[]) };

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<ComputeSkinningShaderConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::all())
            .build();

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(desc_set_layouts)
            .push_constant_ranges(&[push_constant_range])
            .build();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&layout_create_info, None) }.unwrap();

        let entry_point = ShaderEntryPoint {
            entry_point: String::from("main_cs"),
        };

        let shader_module: vk::ShaderModule = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(shader_file.data.as_slice()),
                    None,
                )
                .expect("Shader module creation error")
        };

        let shader_module_set = {
            let compute_module = shader_module;
            let compute_entry_point = CString::new(entry_point.entry_point).unwrap();
            ShaderModules::compute(compute_module, compute_entry_point)
        };

        let shader_stage_create_info = {
            vk::PipelineShaderStageCreateInfo {
                module: shader_module_set.compute_module.unwrap(),
                p_name: (*shader_module_set.compute_entry_point.as_ref().unwrap()).as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            }
        };

        let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage_create_info)
            .layout(pipeline_layout);

        let compute_pipeline_create_info = compute_pipeline_info.build();

        let compute_pipelines = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[compute_pipeline_create_info],
                None,
            )
        };

        let compute_pipeline = compute_pipelines.unwrap()[0];

        ComputeSkinningPass {
            pipeline_layout,
            bone_pose_buffer,
            desc_set_layout,
            compute_pipeline,
            descriptor_sets,
            shader_module,
            object_to_world: Mat4::IDENTITY,
        }
    }

    pub fn update(&self) {}

    pub fn gpu_setup(&self, _device: &Device, _command_buffer: &vk::CommandBuffer) {}

    pub fn dispatch(
        &self,
        shader_constants: &shared::ComputeSkinningShaderConstants,
        device: &Device,
        command_buffer: &vk::CommandBuffer,
    ) {
        unsafe {
            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &self.descriptor_sets[..],
                &[],
            );

            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline,
            );

            let push_constants = shader_constants;
            device.cmd_push_constants(
                *command_buffer,
                self.pipeline_layout,
                ash::vk::ShaderStageFlags::all(),
                0,
                any_as_u8_slice(push_constants),
            );

            let group_count = 1;
            device.cmd_dispatch(*command_buffer, group_count, group_count, group_count);
        }
    }

    pub fn destroy(&self, device: &Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_shader_module(self.shader_module, None);
            self.bone_pose_buffer.destroy(allocator);
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}
