#![allow(dead_code, unused)]
#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr),
    register_attr(spirv)
)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

use shared::glam::{UVec3};
//use shared::{SkeletonPoseGPU, SkinningConstants, SkinningVertexBuffer0, SkinningVertexBuffer1, glam::{Mat4, Quat, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles}};

const GRID_SIZE: usize = 8;

#[spirv(compute(threads(64, 0, 0)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    // #[spirv(push_constant)] skinning_constants: &SkinningConstants,
    // //skeleton_gpu_buffer: &SkeletonGPUBuffer,
    // #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] bone_pose_buffer_0: &SkeletonPoseGPU,
    // #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bone_pose_buffer_1: &SkeletonPoseGPU,
    // vertex_buffer_0: &mut SkinningVertexBuffer0,
    // vertex_buffer_bind: &mut SkinningVertexBuffer1,
    // vertex_buffer_dyn: &mut SkinningVertexBuffer1,
) {
    //let index = id.x as usize + id.y as usize * GRID_SIZE + id.z as usize * GRID_SIZE * GRID_SIZE;
    // vertex_buffer.bone_positions[index + 0] = 1.0;
    // vertex_buffer.bone_positions[index + 1] = 2.0;
    // vertex_buffer.bone_positions[index + 2] = 3.0;
    // vertex_buffer.bone_positions[index + 3] = 4.0;



    // let index = id.x as usize;
    // let size_of_f32 = 4;

    // let vertex_stride_bones = 2 * size_of_f32;
    // let vertex_stride_bind = 3 * size_of_f32;
    // let vertex_stride_dyn = 3 * size_of_f32;

    // let pos_0 = bone_pose_buffer_0.joint_poses[index].pos;
    // let pos_1 = bone_pose_buffer_1.joint_poses[index].pos;

    // let rot_0 = bone_pose_buffer_0.joint_poses[index].rot;
    // let rot_1 = bone_pose_buffer_1.joint_poses[index].rot;

    // let scale_0 = bone_pose_buffer_0.joint_poses[index].scale;
    // let scale_1 = bone_pose_buffer_1.joint_poses[index].scale;

    // let t = skinning_constants.t;

    // let pos_final = Vec3::from_slice(&pos_0).lerp(Vec3::from_slice(&pos_1), t);
    // let rot_final = Quat::from_slice(&rot_0).slerp(Quat::from_slice(&rot_1), t);
    // let scale_final = Vec3::from_slice(&scale_0).lerp(Vec3::from_slice(&scale_1), t);

    // let model = Mat4::from_scale(scale_final)
    //     * Mat4::from_quat(rot_final)
    //     * Mat4::from_translation(pos_final);

    // let i = index * vertex_stride_dyn;
    // let pos_bind_space = Vec4::from((
    //     Vec3::from_slice(&vertex_buffer_bind.pos[i..i + vertex_stride_dyn]),
    //     1.0,
    // ));
    // let norm_bind_space = Vec4::from((
    //     Vec3::from_slice(&vertex_buffer_bind.norm[i..i + vertex_stride_dyn]),
    //     0.0,
    // ));
    // let pos_ws = model * pos_bind_space;
    // let norm_ws = model * norm_bind_space;
    // let uv = Vec2::from_slice(&vertex_buffer_bind.uv[i..i + vertex_stride_dyn]);

    // let pos_index = index * 3 * size_of_f32;
    // vertex_buffer_dyn.pos[pos_index + size_of_f32 * 0] = pos_ws.x;
    // vertex_buffer_dyn.pos[pos_index + size_of_f32 * 1] = pos_ws.y;
    // vertex_buffer_dyn.pos[pos_index + size_of_f32 * 2] = pos_ws.z;

    // let norm_index = index * 3 * size_of_f32;
    // vertex_buffer_dyn.norm[norm_index + size_of_f32 * 0] = norm_ws.x;
    // vertex_buffer_dyn.norm[norm_index + size_of_f32 * 1] = norm_ws.y;
    // vertex_buffer_dyn.norm[norm_index + size_of_f32 * 2] = norm_ws.z;

    // let uv_index = index * 2 * size_of_f32;
    // vertex_buffer_dyn.uv[uv_index + size_of_f32 * 0] = uv.x;
    // vertex_buffer_dyn.uv[uv_index + size_of_f32 * 1] = uv.y;
}
