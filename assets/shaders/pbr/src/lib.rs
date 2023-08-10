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
    glam::{IVec4, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles},
    MeshShaderConstants, SkinningVertexBuffer1,
};

#[spirv(vertex)]
pub fn main_vs(
    in_pos: Vec3,
    in_normal: Vec3,
    in_texcoord: Vec2,
    in_colour: Vec4,
    #[spirv(position, invariant)] out_pos: &mut Vec4,
    out_position_ws: &mut Vec4,
    out_normal: &mut Vec3,
    out_texcoord: &mut Vec2,
    out_colour: &mut Vec4,
    #[spirv(push_constant)] constants: &MeshShaderConstants,
) {
    let mut pos = in_pos;
    pos.y *= (constants.time * 2.0).sin() * 0.2 + 0.9;

    let object_to_world = Mat4::from_cols_array(&constants.object_to_world);
    *out_position_ws = object_to_world /* * pose0 */ * Vec4::from((pos, 0.0));

    *out_normal = (object_to_world * Vec4::from((in_normal, 0.0))).xyz();
    *out_texcoord = in_texcoord;
    *out_colour = in_colour;
    *out_pos = Mat4::from_cols_array(&constants.world_to_screen) * Vec4::from((pos, 1.0));
}

#[spirv(fragment)]
pub fn main_fs(
    in_position_ws: Vec4,
    in_normal: Vec3,
    in_texcoord: Vec2,
    in_colour: Vec4,
    out_frag_colour: &mut Vec4,
    #[spirv(push_constant)] constants: &MeshShaderConstants,
) {
    let l = Vec3::new(0.5, 0.5, 0.5).normalize();
    let n_dot_l = in_normal.dot(l);
    *out_frag_colour = Vec4::new(in_position_ws.x, in_position_ws.z, 0.0, 1.0);
    // *out_frag_colour = Vec4::from((in_normal, 1.0));
    // *out_frag_colour *= in_colour;
    // *out_frag_colour *= Vec4::from((in_texcoord, 1.0, 1.0));
    *out_frag_colour *= 5.0 * (constants.time.sin() * 0.5 + 0.5);
    //let colour = Vec3::new(constants.color[0], constants.color[1], constants.color[2]);
    //*out_frag_colour = Vec4::from((in_colour.xyz() * colour * n_dot_l, constants.color[3]));
}
