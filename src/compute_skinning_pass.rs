use glam::*;
//use shared::SkeletonPoseGPU;
use shared::SkinningConstants;
use shared::SKINNING_COMPUTE_GROUP_SIZE;
use std::default::Default;
use std::ffi::CString;
use std::rc::Weak;

use ash::{vk, Device};

use crate::game::SpvFile;
use crate::mesh::*;
use crate::render_pass_common::*;
use crate::vulkan_base::*;

#[allow(dead_code)]
pub struct ComputeSkinningPass {
    pub pipeline_layout: vk::PipelineLayout,
    //pub bone_pose_gpu_buffer_0: VkBuffer,
    //pub bone_pose_gpu_buffer_1: VkBuffer,
    pub desc_set_layout_0: vk::DescriptorSetLayout,
    pub desc_set_layout_1: vk::DescriptorSetLayout,
    pub compute_pipeline: vk::Pipeline,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub shader_module: vk::ShaderModule,
    pub mesh: Weak<Mesh<AnimatedVertex>>,
}

impl ComputeSkinningPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        allocator: &vk_mem::Allocator,
        descriptor_pool: &vk::DescriptorPool,
        mesh: Weak<Mesh<AnimatedVertex>>,
        shader_file: &SpvFile,
    ) -> ComputeSkinningPass {
        // let alloc_info = vk_mem::AllocationCreateInfo {
        //     usage: vk_mem::MemoryUsage::CpuToGpu,
        //     flags: vk_mem::AllocationCreateFlags::MAPPED,
        //     ..Default::default()
        // };

        // let bind_pose_data = &mesh.upgrade().unwrap().bone_inv_bind_poses;

        // let bone_pose_buffer_info = vk::BufferCreateInfo {
        //     size: std::mem::size_of::<shared::SkeletonPoseGPU>() as u64,
        //     usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        //     sharing_mode: vk::SharingMode::EXCLUSIVE,
        //     ..Default::default()
        // };

        // let bone_pose_gpu_buffer_0 = VkBuffer::new(allocator, &bone_pose_buffer_info, &alloc_info);
        // let bone_pose_gpu_buffer_1 = VkBuffer::new(allocator, &bone_pose_buffer_info, &alloc_info);

        let desc_layout_bindings_0 = [
            // // SkeletonGPU Buffer
            // vk::DescriptorSetLayoutBinding {
            //     binding: 0,
            //     descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            //     descriptor_count: 1,
            //     stage_flags: vk::ShaderStageFlags::COMPUTE,
            //     ..Default::default()
            // },
            // // Anim pose 0 buffer
            // vk::DescriptorSetLayoutBinding {
            //     binding: 1,
            //     descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            //     descriptor_count: 1,
            //     stage_flags: vk::ShaderStageFlags::COMPUTE,
            //     ..Default::default()
            // },
            // // Anim pose 1 buffer
            // vk::DescriptorSetLayoutBinding {
            //     binding: 2,
            //     descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            //     descriptor_count: 1,
            //     stage_flags: vk::ShaderStageFlags::COMPUTE,
            //     ..Default::default()
            // },
        ];

        let desc_layout_bindings_1 = [
            // // Skinning vertex buffer 0 (bone indices/weights)
            // vk::DescriptorSetLayoutBinding {
            //     binding: 0,
            //     descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            //     descriptor_count: 1,
            //     stage_flags: vk::ShaderStageFlags::COMPUTE,
            //     ..Default::default()
            // },
            // // Skinning vertex buffer 1 ("real" vertex buffer)
            // vk::DescriptorSetLayoutBinding {
            //     binding: 1,
            //     descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            //     descriptor_count: 1,
            //     stage_flags: vk::ShaderStageFlags::COMPUTE,
            //     ..Default::default()
            // },
        ];

        let descriptor_info_0 =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings_0);

        let descriptor_info_1 =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings_1);

        let desc_set_layout_0 =
            unsafe { device.create_descriptor_set_layout(&descriptor_info_0, None) }.unwrap();

        let desc_set_layout_1 =
            unsafe { device.create_descriptor_set_layout(&descriptor_info_1, None) }.unwrap();

        let desc_set_layouts = &[desc_set_layout_0, desc_set_layout_1];

        let descriptor_sets = {
            let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(*descriptor_pool)
                .set_layouts(desc_set_layouts);

            unsafe { device.allocate_descriptor_sets(&desc_alloc_info) }.unwrap()
        };

        // TODO:
        let write_desc_sets = [];
        unsafe { device.update_descriptor_sets(&write_desc_sets, &[]) };

        let push_constant_range = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<SkinningConstants>() as u32)
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
            //bone_pose_gpu_buffer_0,
            //bone_pose_gpu_buffer_1,
            desc_set_layout_0,
            desc_set_layout_1,
            compute_pipeline,
            descriptor_sets,
            shader_module,
            mesh,
        }
    }

    pub fn update(&self) {
        let mesh = self.mesh.upgrade().unwrap();
        //let anim_clip = mesh.anim_clips[mesh.active_clip as usize];

        // anim_clip.frame_index += 1;

        // let frame_index_0 = anim_clip.frame_index;
        // let frame_index_1 = (anim_clip.frame_index + 1) % anim_clip.frame_count;

        // let joint_pose_0 = SkeletonPoseGPU::default();
        // let joint_pose_1 = SkeletonPoseGPU::default();

        // if let Some(positions) = anim_clip.positions {
        //     for (i, pos) in positions.iter().enumerate() {
        //         joint_pose_0.joint_poses[i].pos = [pos.x, pos.y, pos.z, 0.0];
        //     }
        // } else {
        //     for i in 0..anim_clip.joint_count {
        //         joint_pose_0.joint_poses[i].pos = [0.0; 4];
        //     }
        // }
        // if let Some(rotations) = anim_clip.rotations {
        //     for (i, rot) in rotations.iter().enumerate() {
        //         joint_pose_0.joint_poses[i].rot = [rot.x, rot.y, rot.z, rot.w];
        //     }
        // } else {
        //     for i in 0..anim_clip.joint_count {
        //         joint_pose_0.joint_poses[i].rot = [0.0, 0.0, 0.0, 1.0];
        //     }
        // }
        // if let Some(scales) = anim_clip.scales {
        //     for (i, scale) in scales.iter().enumerate() {
        //         joint_pose_0.joint_poses[i].scale = [scale.x, scale.y, scale.z, 0.0];
        //     }
        // } else {
        //     for i in 0..anim_clip.joint_count {
        //         joint_pose_0.joint_poses[i].scale = [1.0, 1.0, 1.0, 0.0];
        //     }
        // }

        //self.bind_pose_data[]

        //self.bone_pose_gpu_buffer_0
        //    .copy_from_slice(&joint_pose_0.joint_poses[..], 0);
        //self.bone_pose_gpu_buffer_1
        //    .copy_from_slice(&joint_pose_1.joint_poses[..], 0);
    }

    pub fn gpu_setup(&self, _device: &Device, _command_buffer: &vk::CommandBuffer) {}

    pub fn dispatch(
        &self,
        skinning_constants: &shared::SkinningConstants,
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

            device.cmd_push_constants(
                *command_buffer,
                self.pipeline_layout,
                ash::vk::ShaderStageFlags::all(),
                0,
                any_as_u8_slice(skinning_constants),
            );

            //let group_count = 1; // num verts / group size
            //let group_count = self.mesh.upgrade().unwrap().submeshes[0].vertices.len() as u32
            //    / SKINNING_COMPUTE_GROUP_SIZE as u32;
            //device.cmd_dispatch(*command_buffer, group_count, group_count, group_count);
        }
    }

    pub fn destroy(&self, device: &Device, allocator: &vk_mem::Allocator) {
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_shader_module(self.shader_module, None);
            //self.bone_pose_gpu_buffer_0.destroy(allocator);
            //self.bone_pose_gpu_buffer_1.destroy(allocator);
            device.destroy_descriptor_set_layout(self.desc_set_layout_0, None);
            device.destroy_descriptor_set_layout(self.desc_set_layout_1, None);
        }
    }
}
