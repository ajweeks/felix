#![cfg_attr(target_arch = "spirv", no_std, feature(lang_items))]

use core::f32::consts::PI;
use glam::{vec3, Vec3};

pub use spirv_std::glam;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
pub struct MeshShaderConstants {
    pub object_to_world: [f32; 16],
    pub world_to_screen: [f32; 16],
    pub color: [f32; 4],
    pub time: f32,
}

pub const MAX_NUM_BONES: usize = 4;

#[derive(Copy, Clone /*, Zeroable, Pod*/)]
#[repr(C)]
pub struct ComputeSkinningShaderConstants {
    pub bone_poses: [f32; 16 * MAX_NUM_BONES],
}

#[derive(Copy, Clone /*, Zeroable, Pod*/)]
#[repr(C)]
pub struct BonePoseBuffer {
    pub bone_poses: [f32; 16 * MAX_NUM_BONES],
}

#[derive(Copy, Clone /*, Zeroable, Pod*/)]
#[repr(C)]
pub struct SkinningVertexBuffer {
    pub bone_positions: [f32; 4 * MAX_NUM_BONES],
}

pub fn saturate(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

pub fn pow(v: Vec3, power: f32) -> Vec3 {
    vec3(v.x.powf(power), v.y.powf(power), v.z.powf(power))
}

pub fn exp(v: Vec3) -> Vec3 {
    vec3(v.x.exp(), v.y.exp(), v.z.exp())
}

/// Based on: <https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/>
pub fn acos_approx(v: f32) -> f32 {
    let x = v.abs();
    let mut res = -0.155972 * x + 1.56467; // p(x)
    res *= (1.0f32 - x).sqrt();

    if v >= 0.0 {
        res
    } else {
        PI - res
    }
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    // Scale, bias and saturate x to 0..1 range
    let x = saturate((x - edge0) / (edge1 - edge0));
    // Evaluate polynomial
    x * x * (3.0 - 2.0 * x)
}
