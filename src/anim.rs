use glam::{Mat4, Quat, Vec3, Vec4};
use gltf::animation::Property;
//use shared::JointPose;
use std::rc::Rc;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Joint {
    // TODO: 4x3
    pub inv_bind_pose: Mat4,
    // pub name: str,
    pub parent_index: u8,
}

// #[derive(Clone, Copy)]
pub struct Skeleton {
    pub joint_count: usize,
    pub joints: Vec<Joint>,
}

// #[derive(Clone, Copy)]
//pub struct SkeletonPose {
//    pub skeleton: Rc<Skeleton>,
//    pub joint_poses: Vec<JointPose>,
//}

pub struct AnimClipCPU {
    // Runtime state
    pub frame_index: usize,

    // Serialized state
    pub channel_count: usize,
    pub frame_count: usize,
    pub joint_count: usize,
    pub positions: Option<Vec<Vec3>>,
    pub rotations: Option<Vec<Quat>>,
    pub scales: Option<Vec<Vec3>>,
}

impl Default for AnimClipCPU {
    fn default() -> AnimClipCPU {
        AnimClipCPU {
            channel_count: 0,
            frame_index: 0,
            frame_count: 0,
            joint_count: 0,
            positions: None,
            rotations: None,
            scales: None,
        }
    }
}

impl AnimClipCPU {
    // pub fn add_value3(&mut self, val: [f32; 4], property: Property) {
    //     match property {
    //         Property::Translation => self.positions.as_mut().unwrap().push(val),
    //         Property::Scale => self.scales.as_mut().unwrap().push(val),
    //         _ => panic!("Unexpected value"),
    //     }
    // }

    pub fn add_vec3(&mut self, val: Vec3, property: Property) {
        match property {
            Property::Translation => self.positions.as_mut().unwrap().push(val),
            Property::Scale => self.scales.as_mut().unwrap().push(val),
            _ => panic!("Unexpected value"),
        }
    }
    
    //pub fn add_vec4(&mut self, val: Vec4, property: Property) {
    //    match property {
    //        Property::Translation => self.positions.as_mut().unwrap().push(val),
    //        Property::Scale => self.scales.as_mut().unwrap().push(val),
    //        _ => panic!("Unexpected value"),
    //    }
    //}

    pub fn add_quat(&mut self, val: Quat, property: Property) {
        match property {
            Property::Rotation => self.rotations.as_mut().unwrap().push(val),
            _ => panic!("Unexpected value"),
        }
    }
}

// pub struct AnimationCollection {
//     pub animation_clips: Vec<AnimationClip>,
//     pub clip_index: usize,
// }

// impl AnimationCollection {
//     pub fn new() -> AnimationCollection {
//         AnimationCollection {
//             animation_clips: Vec::new(),
//             clip_index: 0,
//         }
//     }
// }

// pub struct Animation
// {
//     pub mesh: Weak<Mesh>,
//     pub anim_frame_data: Vec<Vec<Bone>>,
// }
