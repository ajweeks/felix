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

use shared::{
    glam::{Mat4, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles},
    BonePoseBuffer, ComputeSkinningShaderConstants, SkinningVertexBuffer,
};

const GRID_SIZE: usize = 8;
const GROUP_SIZE: usize = 8;

#[spirv(compute(threads(8, 8, 8)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] constants: &ComputeSkinningShaderConstants,
    bone_pose_buffer: BonePoseBuffer,
    vertex_buffer: &mut SkinningVertexBuffer,
) {
    let index = id.x as usize + id.y as usize * GRID_SIZE + id.z as usize * GRID_SIZE * GRID_SIZE;
    // vertex_buffer.bone_positions[index + 0] = 1.0;
    // vertex_buffer.bone_positions[index + 1] = 2.0;
    // vertex_buffer.bone_positions[index + 2] = 3.0;
    // vertex_buffer.bone_positions[index + 3] = 4.0;
}
