use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::{fs, io, slice};

use byteorder::ByteOrder;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use glam::*;
use gltf::animation::Property;
use gltf::Glb;
use gltf::Gltf;

#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub colour: [f32; 4],
}

#[derive(Clone, Copy)]
pub struct Bone {
    pub transform: Mat4,
}

impl From<Mat4> for Bone {
    fn from(mat: Mat4) -> Bone {
        Bone { transform: mat }
    }
}

pub struct Animation {
    pub bone_animations: Vec<BoneAnimation>,
    pub frame_index: usize,
    pub frame_count: usize,
}

pub struct BoneAnimation {
    pub channel_count: usize,
    pub positions: Option<Vec<[f32; 3]>>,
    pub rotations: Option<Vec<[f32; 4]>>,
    pub scales: Option<Vec<[f32; 3]>>,
}

impl Default for BoneAnimation {
    fn default() -> BoneAnimation {
        BoneAnimation {
            channel_count: 0,
            positions: None,
            rotations: None,
            scales: None,
        }
    }
}

impl BoneAnimation {
    pub fn add_value3(&mut self, val: [f32; 3], property: Property) {
        match property {
            Property::Translation => self.positions.as_mut().unwrap().push(val),
            Property::Scale => self.scales.as_mut().unwrap().push(val),
            _ => panic!("Unexpected value"),
        }
    }

    pub fn add_value4(&mut self, val: [f32; 4], property: Property) {
        match property {
            Property::Rotation => self.rotations.as_mut().unwrap().push(val),
            _ => panic!("Unexpected value"),
        }
    }
}

pub struct Mesh {
    pub submeshes: Vec<SubMesh>,
    pub bone_poses: Vec<Bone>,
    pub anims: Vec<Animation>,
}

impl Mesh {
    pub fn load_gltf_mesh(mesh_path: &str) -> Mesh {
        println!("Loading mesh {}...", mesh_path);

        let gltf = Gltf::open(format!("assets/meshes/{}", mesh_path)).expect("Failed to load mesh");
        let glb = Glb::from_reader(
            fs::File::open(format!("assets/meshes/{}", mesh_path)).expect("Failed to load mesh"),
        )
        .expect("Failed to load mesh");

        // println!("Mesh has {} scene(s)", gltf.scenes().count());

        let mut submeshes = Vec::new();

        let mut bone_poses = Vec::new();

        for scene in gltf.scenes() {
            //println!(
            //    "Scene {} has {} node(s)",
            //    scene.index(),
            //    scene.nodes().count()
            //);
            for node in scene.nodes() {
                // TODO: unmut
                let mut bytes = Cursor::new(glb.bin.as_ref().unwrap());

                println!("Skin: {}", node.skin().is_some());
                if let Some(skin) = node.skin() {
                    // if let Some(accessor) = skin.inverse_bind_matrices() {
                    //     let view = accessor.view().unwrap();
                    //     const STRIDE: usize = 16 * 4;
                    //     bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                    //     let mut element_bytes = [0; STRIDE];
                    //     for _ in 0..accessor.count() {
                    //         bytes.read_exact(&mut element_bytes).unwrap();
                    //         let mut mat_values = [[0.0; 4]; 4];
                    //         let mut j = 0;
                    //         for i in 0..4 {
                    //             let x = LittleEndian::read_f32(&element_bytes[(j + 0)..(j + 4)]);
                    //             let y = LittleEndian::read_f32(&element_bytes[(j + 4)..(j + 8)]);
                    //             let z = LittleEndian::read_f32(&element_bytes[(j + 8)..(j + 12)]);
                    //             let w = LittleEndian::read_f32(&element_bytes[(j + 12)..(j + 16)]);
                    //             mat_values[i] = [x, y, z, w];
                    //             j += 16;
                    //         }
                    //         bone_poses.push(Bone::from(Mat4::from_cols(
                    //             Vec4::from(mat_values[0]),
                    //             Vec4::from(mat_values[1]),
                    //             Vec4::from(mat_values[2]),
                    //             Vec4::from(mat_values[3]),
                    //         )));
                    //         println!(
                    //             "bone poses: {}",
                    //             &bone_poses[bone_poses.len() - 1].transform
                    //         );
                    //     }
                    // }

                    println!("Bone count: {}\nBone names:", skin.joints().count());
                    for joint in skin.joints() {
                        let matrix = joint.transform().matrix();

                        println!(
                            "\t{}:\n\t{:?}\n\t{:?}\n\t{:?}\n\t{:?}",
                            joint.name().unwrap(),
                            matrix[0],
                            matrix[1],
                            matrix[2],
                            matrix[2]
                        );

                        bone_poses.push(Bone::from(Mat4::from_cols(
                            Vec4::from(matrix[0]),
                            Vec4::from(matrix[1]),
                            Vec4::from(matrix[2]),
                            Vec4::from(matrix[3]),
                        )));
                    }
                }

                if let Some(mesh) = node.mesh() {
                    //println!(
                    //    "Mesh: {} has {} primitive(s)",
                    //    mesh.name().unwrap_or("Empty"),
                    //    mesh.primitives().count()
                    //);

                    for primitive in mesh.primitives() {
                        //let indices = primitive.indices().unwrap();
                        //let bb = primitive.bounding_box();
                        //println!("Index count: {}", indices.count());
                        //println!("Bounding box {:?}", bb);
                        // println!("Mesh: has {} attributes", primitive.attributes().count());

                        let vertex_count =
                            primitive.get(&gltf::Semantic::Positions).unwrap().count();

                        // println!("Mesh has {} verts", vertex_count);

                        let parse_float4_attribute =
                            |bytes: &mut Cursor<&Cow<[u8]>>,
                             semantic: &gltf::Semantic,
                             default_value: [f32; 4]| {
                                let mut output = Vec::new();
                                output.reserve(vertex_count);

                                if let Some(accessor) = primitive.get(semantic) {
                                    let view = accessor.view().unwrap();
                                    const STRIDE: usize = 4 * 4;
                                    bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                                    let mut element_bytes = [0; STRIDE];
                                    for _ in 0..accessor.count() {
                                        bytes.read_exact(&mut element_bytes).unwrap();
                                        let x = LittleEndian::read_f32(&element_bytes[0..4]);
                                        let y = LittleEndian::read_f32(&element_bytes[4..8]);
                                        let z = LittleEndian::read_f32(&element_bytes[8..12]);
                                        let w = LittleEndian::read_f32(&element_bytes[12..16]);
                                        output.push([x, y, z, w]);
                                        //println!("{}: {:?}, len: {}", i, output[output.len() - 1], (x * x + y * y + z * z).sqrt());
                                    }
                                } else {
                                    output.resize(vertex_count, default_value);
                                }

                                output
                            };

                        let parse_float3_attribute =
                            |bytes: &mut Cursor<&Cow<[u8]>>,
                             semantic: &gltf::Semantic,
                             default_value: [f32; 3]| {
                                let mut output = Vec::new();
                                output.reserve(vertex_count);

                                if let Some(accessor) = primitive.get(semantic) {
                                    let view = accessor.view().unwrap();
                                    const STRIDE: usize = 3 * 4;
                                    bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                                    let mut element_bytes = [0; STRIDE];
                                    for _ in 0..accessor.count() {
                                        bytes.read_exact(&mut element_bytes).unwrap();
                                        let x = LittleEndian::read_f32(&element_bytes[0..4]);
                                        let y = LittleEndian::read_f32(&element_bytes[4..8]);
                                        let z = LittleEndian::read_f32(&element_bytes[8..12]);
                                        output.push([x, y, z]);
                                        //println!("{}: {:?}, len: {}", i, output[output.len() - 1], (x * x + y * y + z * z).sqrt());
                                    }
                                } else {
                                    output.resize(vertex_count, default_value);
                                }

                                output
                            };

                        let parse_float2_attribute =
                            |bytes: &mut Cursor<&Cow<[u8]>>,
                             semantic: &gltf::Semantic,
                             default_value: [f32; 2]| {
                                let mut output = Vec::new();
                                output.reserve(vertex_count);
                                if let Some(accessor) = primitive.get(semantic) {
                                    let view = accessor.view().unwrap();
                                    const STRIDE: usize = 2 * 4;
                                    bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                                    let mut element_bytes = [0; STRIDE];
                                    for _ in 0..accessor.count() {
                                        bytes.read_exact(&mut element_bytes).unwrap();
                                        let x = LittleEndian::read_f32(&element_bytes[0..4]);
                                        let y = LittleEndian::read_f32(&element_bytes[4..8]);
                                        output.push([x, y]);
                                        //println!("{}: {:?}", i, output[output.len() - 1]);
                                    }
                                } else {
                                    output.resize(vertex_count, default_value);
                                }

                                output
                            };

                        let parse_indices = |bytes: &mut Cursor<&Cow<[u8]>>| {
                            let mut indices: Vec<u32> = Vec::new();
                            let indices_iter = primitive.indices().unwrap();
                            let index_count = indices_iter.count();
                            indices.reserve(index_count);
                            let index_stride = indices_iter.size();
                            for _ in 0..index_count {
                                let index: u32 = match index_stride {
                                    1 => bytes.read_u8().unwrap() as u32,
                                    2 => bytes.read_u16::<LittleEndian>().unwrap() as u32,
                                    4 => bytes.read_u32::<LittleEndian>().unwrap() as u32,
                                    _ => 0,
                                };
                                indices.push(index);

                                // println!("{}: {:?}", i, indices[indices.len() - 1]);
                            }
                            indices
                        };

                        let parse_bone_indices = |bytes: &mut Cursor<&Cow<[u8]>>| {
                            let mut output = Vec::new();
                            println!(
                                "Joints: {}",
                                primitive.get(&gltf::Semantic::Joints(0)).is_some()
                            );
                            if let Some(accessor) = primitive.get(&gltf::Semantic::Joints(0)) {
                                output.reserve(vertex_count);
                                let view = accessor.view().unwrap();
                                const STRIDE: usize = 4;
                                bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();

                                println!("parse_bones > accessor.count: {}", accessor.count());

                                let mut element_bytes = [0; STRIDE];
                                for _ in 0..accessor.count() {
                                    bytes.read_exact(&mut element_bytes).unwrap();
                                    let index = LittleEndian::read_i32(&element_bytes[0..4]);
                                    output.push(index);
                                    // println!("index: {}", index);
                                }
                            }

                            output
                        };

                        let parse_bone_weights = |bytes: &mut Cursor<&Cow<[u8]>>| {
                            let mut output = Vec::new();
                            println!(
                                "Weights: {}",
                                primitive.get(&gltf::Semantic::Weights(0)).is_some()
                            );
                            if let Some(accessor) = primitive.get(&gltf::Semantic::Weights(0)) {
                                output.reserve(vertex_count);
                                let view = accessor.view().unwrap();
                                const STRIDE: usize = 4;
                                bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();

                                println!(
                                    "parse_bone_weights > accessor.count: {}",
                                    accessor.count()
                                );

                                let mut element_bytes = [0; STRIDE];
                                for _ in 0..accessor.count() {
                                    bytes.read_exact(&mut element_bytes).unwrap();
                                    let weight = LittleEndian::read_f32(&element_bytes[0..4]);
                                    output.push(weight);
                                    // println!("weight: {}", weight);
                                }
                            }

                            output
                        };

                        let positions = parse_float3_attribute(
                            &mut bytes,
                            &gltf::Semantic::Positions,
                            [0.0, 0.0, 0.0],
                        );
                        let normals = parse_float3_attribute(
                            &mut bytes,
                            &gltf::Semantic::Normals,
                            [0.0, 1.0, 0.0],
                        );
                        let texcoords = parse_float2_attribute(
                            &mut bytes,
                            &gltf::Semantic::TexCoords(0),
                            [0.0, 0.0],
                        );
                        let colours = parse_float4_attribute(
                            &mut bytes,
                            &gltf::Semantic::Colors(0),
                            [1.0; 4],
                        );
                        let indices = parse_indices(&mut bytes);

                        let mut vertices: Vec<Vertex> = Vec::new();
                        vertices.reserve(vertex_count);
                        for i in 0..vertex_count {
                            vertices.push(Vertex {
                                pos: positions[i],
                                normal: normals[i],
                                uv: texcoords[i],
                                colour: colours[i],
                            });
                        }

                        let bone_indices = parse_bone_indices(&mut bytes);
                        let bone_weights = parse_bone_weights(&mut bytes);

                        submeshes.push(SubMesh {
                            vertices,
                            indices,
                            bone_indices,
                            bone_weights,
                        })

                        //for attr in primitive.attributes() {
                        //    println!("\t{:?}: {}", attr.0, attr.1.count());
                        //}
                    }
                }
            }
        }

        let mut bytes = Cursor::new(glb.bin.as_ref().unwrap());
        let mut anims = Vec::new();
        for anim in gltf.animations() {
            let mut bone_animation_tracks: HashMap<String, BoneAnimation> = HashMap::new();

            let mut channel_lookup = Vec::new();

            let mut register_property = |property, bone| {
                let bone_anim = bone_animation_tracks.entry(bone).or_default();
                match property {
                    Property::Translation => {
                        if bone_anim.positions.is_none() {
                            bone_anim.positions = Some(Vec::new());
                            bone_anim.channel_count += 1;
                        }
                    }
                    Property::Rotation => {
                        if bone_anim.rotations.is_none() {
                            bone_anim.rotations = Some(Vec::new());
                            bone_anim.channel_count += 1;
                        }
                    }
                    Property::Scale => {
                        if bone_anim.scales.is_none() {
                            bone_anim.scales = Some(Vec::new());
                            bone_anim.channel_count += 1;
                        }
                    }
                    _ => {}
                }
            };

            for channel in anim.channels() {
                let target = channel.target();

                let name = target.node().name().unwrap().to_owned();

                channel_lookup.push((target.property(), name.clone()));
                register_property(target.property(), name);
            }

            let frame_count = anim.samplers().next().unwrap().output().count();

            for (sampler_index, sampler) in anim.samplers().enumerate() {
                const VEC3_STRIDE: usize = 3 * 4;
                const VEC4_STRIDE: usize = 4 * 4;

                let accessor = sampler.output();
                let view = accessor.view().unwrap();
                bytes.seek(SeekFrom::Start((view.offset()) as u64)).unwrap();
                let mut vec3_element_bytes = [0; VEC3_STRIDE];
                let mut vec4_element_bytes = [0; VEC4_STRIDE];

                let (sample_property, sample_bone_name) =
                    channel_lookup.get(sampler_index).unwrap();

                if let Some(bone_anim) = bone_animation_tracks.get_mut(sample_bone_name) {
                    for _ in 0..accessor.count() {
                        match *sample_property {
                            Property::Translation | Property::Scale => {
                                bytes.read_exact(&mut vec3_element_bytes).unwrap();
                                let x = LittleEndian::read_f32(&vec3_element_bytes[0..4]);
                                let y = LittleEndian::read_f32(&vec3_element_bytes[4..8]);
                                let z = LittleEndian::read_f32(&vec3_element_bytes[8..12]);
                                bone_anim.add_value3([x, y, z], *sample_property);
                                println!("{:?}, {:?}", *sample_property, [x, y, z]);
                            }
                            Property::Rotation => {
                                bytes.read_exact(&mut vec4_element_bytes).unwrap();
                                let x = LittleEndian::read_f32(&vec4_element_bytes[0..4]);
                                let y = LittleEndian::read_f32(&vec4_element_bytes[4..8]);
                                let z = LittleEndian::read_f32(&vec4_element_bytes[8..12]);
                                let w = LittleEndian::read_f32(&vec4_element_bytes[12..16]);
                                bone_anim.add_value4([x, y, z, w], *sample_property);
                                println!("{:?}, {:?}", *sample_property, [x, y, z, w]);
                            }
                            _ => {}
                        }
                    }
                }
            }

            anims.push(Animation {
                bone_animations: bone_animation_tracks.into_values().collect(),
                frame_index: 0,
                frame_count,
            });
        }

        Mesh {
            submeshes,
            bone_poses,
            anims: Vec::new(),
        }
    }

    pub fn read_spv<R: io::Read + io::Seek>(x: &mut R) -> io::Result<Vec<u32>> {
        let size = x.seek(io::SeekFrom::End(0))?;
        if size % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input length not divisible by 4",
            ));
        }
        if size > usize::max_value() as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
        }
        let words = (size / 4) as usize;
        let mut result = Vec::<u32>::with_capacity(words);
        x.seek(io::SeekFrom::Start(0))?;
        unsafe {
            x.read_exact(slice::from_raw_parts_mut(
                result.as_mut_ptr() as *mut u8,
                words * 4,
            ))?;
            result.set_len(words);
        }
        const MAGIC_NUMBER: u32 = 0x07230203;
        if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
            for word in &mut result {
                *word = word.swap_bytes();
            }
        }
        if result.is_empty() || result[0] != MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input missing SPIR-V magic number",
            ));
        }
        Ok(result)
    }
}

pub struct SubMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bone_indices: Vec<i32>,
    pub bone_weights: Vec<f32>,
}
