#![cfg_attr(target_arch = "spirv", no_std, feature(lang_items))]

use core::f32::consts::PI;
use glam::{vec3, Vec3};

pub use spirv_std::glam;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use bytemuck::{Pod, Zeroable};

pub const MAX_NUM_JOINTS: usize = 4;
pub const MAX_NUM_VERTS: usize = 65536;

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
pub struct MeshShaderConstants {
    pub object_to_world: [f32; 16],
    pub world_to_screen: [f32; 16],
    pub color: [f32; 4],
    pub time: f32,
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct BoneVertex {
    pub bone_indices: [usize; 4],
    pub bone_weights: [f32; 4],
}

// #[derive(Clone, Copy, Zeroable, Pod)]
// #[repr(C)]
// pub struct JointPose {
//     pub pos: [f32; 4],
//     pub rot: [f32; 4],
//     pub scale: [f32; 4],
// }

// impl Default for JointPose {
//     fn default() -> JointPose {
//         JointPose {
//             pos: [0.0, 0.0, 0.0, 0.0],
//             rot: [0.0, 0.0, 0.0, 0.0],
//             scale: [0.0, 0.0, 0.0, 0.0],
//         }
//     }
// }

// #[derive(Clone, Copy, Zeroable, Pod)]
// #[repr(C)]
// pub struct SkeletonPoseGPU {
//     pub joint_poses: [JointPose; MAX_NUM_JOINTS],
// }

// impl Default for SkeletonPoseGPU {
//     fn default() -> SkeletonPoseGPU {
//         SkeletonPoseGPU {
//             joint_poses: [JointPose::default(); MAX_NUM_JOINTS],
//         }
//     }
// }

// #[derive(Clone, Copy, Zeroable, Pod)]
// #[repr(C)]
// pub struct BonePose {
//     pub inv_bind_pose: [f32; 16],
// }

// impl From<[f32; 16]> for BonePose {
//     fn from(inv_bind_pose: [f32; 16]) -> BonePose {
//         BonePose { inv_bind_pose }
//     }
// }

// #[derive(Copy, Clone, Zeroable, Pod)]
// #[repr(C)]
// pub struct SkeletonGPUBuffer {
//     pub inv_bone_poses: [Bone; MAX_NUM_BONES],
// }

// #[derive(Copy, Clone , Zeroable, Pod)]
// #[repr(C)]
// pub struct PoseGPUBuffer {
//     pub bone_poses: [f32; 16 * MAX_NUM_BONES],
// }

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
pub struct SkinningConstants {
    pub t: f32, // Normalized blend factor [0, 1]
}

#[derive(Copy, Clone)] //, Zeroable, Pod)]
#[repr(C)]
pub struct SkinningVertexBuffer0 {
    pub verts: [BoneVertex; MAX_NUM_VERTS],
    //pub joint_indices: [u32; MAX_NUM_VERTS], // u8 can't be used in storage buffers
    //pub joint_weights: [f32; 3 * MAX_NUM_VERTS], // fourth weight is computed as `1 - (w0 + w1 + w2)`
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
pub struct SkinningVertexBuffer1 {
    pub pos: [f32; 3 * MAX_NUM_JOINTS],
    pub norm: [f32; 3 * MAX_NUM_JOINTS],
    pub uv: [f32; 2 * MAX_NUM_JOINTS],
}

pub const SKINNING_COMPUTE_GROUP_SIZE: usize = 64;

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
